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
import settings
from torch.distributions import Categorical


def get_msg():
  while True:
    s, msg = sub.listen()
    if s:
      print(msg)
      return msg

# Create a new nep node
node = nep.node("Calculator")                                                       
conf = node.hybrid("192.168.3.14")                         
sub = node.new_sub("env", "json", conf)
pub = node.new_pub("calc", "json", conf) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = get_msg().values()

def compute_returns(next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + settings.gamma * R * masks[step]
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

def select_action(probs, epsilon):
  # probs = tensor([5.0393e-03, 3.0475e-01,... 2.6321e-01], device='cuda:0', grad_fn=<SoftmaxBackward0>)
  if random.random() < epsilon:
    print('random is choiced')
    dist = Categorical(probs)
    action = torch.tensor(random.randint(0, action_size-1)).to(device='cuda:0')
    print(action)
    return action, dist
  print('maximum is choiced')
  print(probs.argmax())
  # probs.argmax() = tensor(3, device='cuda:0')
  return probs.argmax(), Categorical(probs)

def trainIters(actor, critic, file_name_end):
  optimizerA = optim.Adam(actor.parameters(), lr=settings.lr)
  optimizerC = optim.Adam(critic.parameters(), lr=settings.lr)

  # Initialize the result data
  cumulated_rewards = []
  succeeds = []
  actor_losses = []
  critic_losses = []
  masks = []
  cols = ['state', 'action', 'reward', 'next_state']
  test_rewards = []
  experiences = pd.DataFrame(columns=cols)

  for nepisode in range(1):
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
    epsilon = 0.0

    for i in range(settings.nsteps):
      state = torch.FloatTensor(state).to(device)
      probs, value = actor(state), critic(state)
      action, dist = select_action(probs, epsilon)

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
        if i == settings.nsteps - 1:
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
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()  # calculate gradient
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

  # TODO When you run test, comment out below two lines
  # torch.save(actor, 'model/actor.pkl')
  # torch.save(critic, 'model/critic.pkl')

  # TODO when you run train and test, switch the below two lines
  # save_results(file_name_end, cumulated_rewards, succeeds, experiences, actor_losses, critic_losses)
  save_results(file_name_end, test_rewards, succeeds=[False for _ in range(settings.nsteps)], experiences=experiences)


if __name__ == '__main__':
  if os.path.exists('model/actor.pkl'):
    actor = torch.load('model/actor.pkl')
    print('Actor Model loaded')
  else:
    actor = Actor(state_size, action_size).to(device)
  if os.path.exists('model/critic.pkl'):
    critic = torch.load('model/critic.pkl')
    print('Critic Model loaded')
  else:
    critic = Critic(state_size, action_size).to(device)
  # trainIters(actor, critic, settings.file_name_end)
  trainIters(actor, critic, 'test')
