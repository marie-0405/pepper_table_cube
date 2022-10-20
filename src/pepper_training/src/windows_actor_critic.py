import json
from turtle import st
import nep
import numpy as np
import os
import random
import pandas as pd
import time

import torch
import torch.optim as optim

from actor import Actor
from critic import Critic
from result_controller import ResultController
from human_data_controller import HumanDataController
import settings

NSTEP = settings.nsteps
FILE_NAME = settings.data_file_name

def get_msg():
  while True:
    s, msg = sub.listen()
    if s:
      print(msg)
      return msg

# Create a new nep node
node = nep.node("Calculator")                                                       
conf = node.hybrid("192.168.0.100")                         
# conf = node.hybrid("192.168.3.14")                         
sub = node.new_sub("env", "json", conf)
pub = node.new_pub("calc", "json", conf) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = get_msg().values()  # TODO NEP
# action_size = 10  # TODO csv
# state_size = 2  # TODO csv

def compute_returns(next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + settings.gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def save_results(result_file_name_end, cumulated_rewards, succeeds, experiences, actor_losses='', critic_losses=''):
  ## Save the results
  result_controller = ResultController(result_file_name_end)
  result_controller.write(cumulated_rewards,
                          succeeds=succeeds,
                          experiences=experiences,
                          actor_losses=actor_losses,
                          critic_losses=critic_losses)

def save_fig(result_file_name_end):
  result_controller = ResultController(result_file_name_end)
  result_controller.plot('cumulated_reward')
  result_controller.plot('actor_loss')
  result_controller.plot('critic_loss')

def load_results(result_file_name_end):
  if os.path.exists('../training_results/results-{}.csv'.format(result_file_name_end)):
    result_controller = ResultController(result_file_name_end)
    cumulated_rewards = result_controller.get_data('cumulated_reward')
    succeeds = result_controller.get_data('succeed')
    actor_losses = result_controller.get_data('actor_loss')
    critic_losses = result_controller.get_data('critic_loss')
    # print('cumulated_reward', cumulated_rewards)
    # print(type(cumulated_rewards))

  else:
    cumulated_rewards = []
    succeeds = []
    actor_losses = []
    critic_losses = []
  return cumulated_rewards, succeeds, actor_losses, critic_losses

def select_action(dist, epsilon):
  # probs = tensor([5.0393e-03, 3.0475e-01,... 2.6321e-01], device='cuda:0', grad_fn=<SoftmaxBackward0>)
  if random.random() < epsilon:
    print('random is chosen')
    action = torch.tensor(random.randint(0, action_size-1)).to(device='cuda:0')
  else:
    print('maximum is chosen')
    action = dist.sample()
  print('action', action)
  return action

def trainIters(actor, critic, result_file_name_end):
  human_data_controller = HumanDataController(FILE_NAME)

  # Initialize the result data
  cumulated_rewards, succeeds, actor_losses, critic_losses = load_results(result_file_name_end)
  # print('succ', succeeds)
  masks = []
  cols = ['state', 'action', 'reward', 'next_state']
  test_rewards = []
  experiences = pd.DataFrame(columns=cols)

  for nepisode in range(len(cumulated_rewards), settings.nepisodes):
    print("Episode: " + str(nepisode))
    cumulated_rewards.append(0)
    log_probs = []
    values = []
    rewards = []
    entropy = 0

    print("state")
    state = get_msg()['state']  # TODO NEP
    # state = human_data_controller.get_data(['Distance1', 'Distance2'], index=0)  # TODO csv
    print(state)
    if nepisode < settings.nepisodes - 1:
      epsilon = settings.epsilon_begin + (settings.epsilon_end - settings.epsilon_begin) * nepisode / settings.nepisodes
    else:
      epsilon = settings.epsilon_end
    print(epsilon)

    for i in range(NSTEP):
      optimizerA.zero_grad()
      optimizerC.zero_grad()
      state = torch.FloatTensor(state).to(device)
      dist, value = actor(state), critic(state)
      action = select_action(dist, epsilon)  # TODO NEP
      # action, log_prob = human_data_controller.get_action(i) # TODO csv

      ## TODO NEP begin
      pub.publish({'action': action.cpu().numpy().tolist()})  # need tolist for sending message as json
      msg = get_msg()
      next_state = msg['next_state']
      reward = msg['reward']
      done = msg['done']
      ## NEP end

      ## TODO csv begin
      # next_state = human_data_controller.get_data(['Distance1', 'Distance2'], index=i)
      # reward, done = human_data_controller.calculate_reward_done(next_state[0], next_state[1])
      ## csv end

  
      log_prob = dist.log_prob(action).unsqueeze(0)
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
      masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
      cumulated_rewards[-1] += reward
      test_rewards.append(reward)
      action_num = action.cpu().numpy()
      experience = pd.DataFrame({'state': ','.join(map(str, state.cpu().numpy())), 'action': action_num, 
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
        if i == NSTEP - 1:
          succeeds.append(False)
        else:
          state = next_state
    # save the model and optimizer by episodes
    # TODO train
    torch.save(actor.state_dict(), 'model/actor.pkl')
    torch.save(critic.state_dict(), 'model/critic.pkl')
    torch.save(optimizerA.state_dict(), 'optimizer/optimizerA.pkl')
    torch.save(optimizerC.state_dict(), 'optimizer/optimizerC.pkl')

    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)

    returns = compute_returns(next_value, rewards, masks)
    log_probs = torch.cat(log_probs)  # TODO NEP
    # log_probs = torch.tensor(log_probs, device=device, requires_grad=True)  # TODO csv
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # Save the losses for plot
    actor_losses.append(actor_loss.detach().item())
    critic_losses.append(critic_loss.detach().item())

    save_results(result_file_name_end, cumulated_rewards, succeeds, experiences, actor_losses, critic_losses)

    # Optimize the weight parameters
    actor_loss.backward()  # calculate gradient
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

  # TODO When you run test, comment out below two lines
  torch.save(actor.state_dict(), 'model/actor.pkl')
  torch.save(critic.state_dict(), 'model/critic.pkl')
  torch.save(optimizerA.state_dict(), 'optimizer/optimizerA.pkl')
  torch.save(optimizerC.state_dict(), 'optimizer/optimizerC.pkl')

  # TODO when you run train and test, switch the below two lines
  save_results(result_file_name_end, cumulated_rewards, succeeds, experiences, actor_losses, critic_losses)
  save_fig(result_file_name_end)
  # save_results(result_file_name_end, test_rewards, succeeds=[False for _ in range(NSTEP)], experiences=experiences)
  # save_fig(result_file_name_end)

if __name__ == '__main__':
  if os.path.exists('model/actor.pkl'):
    actor = Actor(state_size, action_size, 256, 512).to(device)
    actor.load_state_dict(torch.load('model/actor.pkl'))
    actor.train()
    # actor.eval()  # TODO test
    print('Actor Model loaded')

    optimizerA = optim.Adam(actor.parameters(), lr=settings.lr)
    optimizerA.load_state_dict(torch.load('optimizer/optimizerA.pkl'))
    print('Actor Optimizer loaded')
  else:
    actor = Actor(state_size, action_size, 256, 512).to(device)
    optimizerA = optim.Adam(actor.parameters(), lr=settings.lr)
  if os.path.exists('model/critic.pkl'):
    critic = Critic(state_size, action_size, 256, 512).to(device)
    critic.load_state_dict(torch.load('model/critic.pkl'))
    critic.train()
    # critic.eval()  # TODO test
    print('Critic Model loaded')

    optimizerC = optim.Adam(critic.parameters(), lr=settings.lr)
    optimizerC.load_state_dict(torch.load('optimizer/optimizerC.pkl'))
    print('Critic Optimizer loaded')
  else:
    critic = Critic(state_size, action_size, 256, 512).to(device)
    optimizerC = optim.Adam(critic.parameters(), lr=settings.lr)
  trainIters(actor, critic, settings.result_file_name_end)
  # trainIters(actor, critic, 'test1')
