import json
from multiprocessing import resource_sharer
from turtle import st
import nep
import numpy as np
import os
import random
import pandas as pd
import time

import torch
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from actor import Actor
from critic import Critic
from result_controller import ResultController
import settings


def get_msg():
  while True:
    s, msg = sub.listen()
    if s:
      print(msg)
      return msg

# Create a new nep node
node = nep.node("Calculator")                                                       
conf = node.hybrid("192.168.0.101")                         
# conf = node.hybrid("192.168.3.14")                         
sub = node.new_sub("env", "json", conf)
pub = node.new_pub("calc", "json", conf) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = get_msg().values()

def compute_returns(config, next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + config['gamma'] * R * masks[step]
    returns.insert(0, R)
  return returns

def save_results(file_name_end, cumulated_rewards, succeeds, experiences='', actor_losses='', critic_losses=''):
  ## Save the results
  result_controller = ResultController(file_name_end)
  result_controller.write(cumulated_rewards,
                          succeeds=succeeds,
                          experiences=experiences,
                          actor_losses=actor_losses,
                          critic_losses=critic_losses)
  result_controller.plot('reward')
  result_controller.plot('actor_loss')
  result_controller.plot('critic_loss')

def select_action(dist, epsilon):
  # probs = tensor([5.0393e-03, 3.0475e-01,... 2.6321e-01], device='cuda:0', grad_fn=<SoftmaxBackward0>)
  if random.random() < epsilon:
    print('random is chosen')
    action = torch.tensor(random.randint(0, action_size-1)).to(device='cuda:0')
    print(action)
  print('maximum is chosen')
  action = dist.sample()
  print(dist.sample())
  return action, dist

def trainIters(config, options):
  file_name_end = settings.file_name_end
  actor = Actor(state_size, action_size, config['L1'], config['L2']).to(device)
  critic = Critic(state_size, action_size, config['L1'], config['L2']).to(device)

  optimizerA = optim.Adam(actor.parameters(), lr=config['lr'])
  optimizerC = optim.Adam(critic.parameters(), lr=config['lr'])

  # Initialize the result data
  cumulated_rewards = []
  succeeds = []
  actor_losses = []
  critic_losses = []
  masks = []
  cols = ['state', 'action', 'reward', 'next_state']
  test_rewards = []
  experiences = pd.DataFrame(columns=cols)

  for nepisode in range(options['nepisodes']):
    cumulated_rewards.append(0)
    log_probs = []
    values = []
    rewards = []
    entropy = 0

    state = get_msg()['state']
    if nepisode < settings.nepisodes - 1:
      epsilon = settings.epsilon_begin + (settings.epsilon_end - settings.epsilon_begin) * nepisode / settings.nepisodes
    else:
      epsilon = settings.epsilon_end
    print(epsilon)

    for i in range(options['nsteps']):
      optimizerA.zero_grad()
      optimizerC.zero_grad()
      state = torch.FloatTensor(state).to(device)
      dist, value = actor(state), critic(state)
      action = select_action(dist, epsilon)

      pub.publish({'action': action.cpu().numpy().tolist()})  # need tolist for sending message as json
      msg = get_msg()
      next_state = msg['next_state']
      reward = msg['reward']
      done = msg['done']

      log_prob = dist.log_prob(action).unsqueeze(0)
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
      masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
      cumulated_rewards[-1] += reward
      test_rewards.append(reward)
      experience = pd.DataFrame({'state': ','.join(map(str, state.cpu().numpy())), 'action': action.cpu().numpy(), 
                                'reward': reward, 'next_state': ','.join(map(str, next_state))}, index=[0])
      experiences = pd.concat([experiences, experience], ignore_index=True)

      if done:
        print('Iteration: {}, Score: {}'.format(nepisode, i))
        break
      
      # Judgement for End or Not
      if done:
        succeeds.append(True)
        break
      else:
        if i == options['nsteps'] - 1:
          succeeds.append(False)
        else:
          state = next_state

    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)

    returns = compute_returns(next_value, rewards, masks)
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # Save the losses for plot
    actor_losses.append(actor_loss.detach().item())
    critic_losses.append(critic_loss.detach().item())

    # Optimize the weight parameters
    actor_loss.backward()  # calculate gradient
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

  # TODO When you run test, comment out below two lines
  # torch.save(actor, 'model/actor.pkl')
  # torch.save(critic, 'model/critic.pkl')

  # TODO when you run train and test, switch the below two lines
  save_results(file_name_end, cumulated_rewards, succeeds, experiences, actor_losses, critic_losses)
  # save_results(file_name_end, test_rewards, succeeds=[False for _ in range(settings.nsteps)], experiences=experiences)

def main():
  # hyper-paprameter configurations
  config = {
    'L1': tune.choice([16, 32, 128, 256]),
    'L2': tune.choice([16, 32, 128, 256]),
    'lr': tune.qloguniform(1e-6, 5e-1, 1e-6),
    'gamma': tune.quniform(0.1, 1.0, 0.1),
    'critic_loss_fn': tune.choice(['mse_loss', 'smooth_l1_loss']),
  }

  options = {
    'device': 'cuda',
    'nepisodes': 100,
    'nsteps': 30,
  }

  # scheduler
  scheduler = ASHAScheduler(
    metric='cumulated_rewards',  # ??
    mode='max',
    max_t=options['nepisodes'],
    grace_period=10
  )

  # search algorithm
  search_alg = BayesOptSearch(metric='cumulated_rewards', mode='max')
  search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

  # Progress reporter
  reporter = CLIReporter(
    metric_columns=['actor_loss', 'critic_loss', 'cumulated_rewards', 'episode'],
    max_progress_rows=10,
    max_report_frequency=5
  )

  # optimization
  result = tune.run(
    partial(trainIters, options=options),
    resources_per_trial={"cpu": 8, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter
  )

  best_trial = result.get_best_trial("cumulated_rewards", "max", "last")
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final validation cumulated_rewards: {}".format(
      best_trial.last_result["cumulated_rewards"]))

  best_trained_actor = Actor(best_trial.config["L1"], best_trial.config["L2"])
  best_trained_critic = Critic(best_trial.config["L1"], best_trial.config["L2"])


if __name__ == '__main__':
  main()
