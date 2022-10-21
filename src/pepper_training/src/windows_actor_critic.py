import json
from turtle import st
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
from human_env_controller import HumanEnvController
from pepper_env_controller import PepperEnvController
import settings

NSTEP = settings.nsteps
FILE_NAME_END = settings.file_name_end

result_data_controller = ResultController(FILE_NAME_END)
experience_controller = ResultController(FILE_NAME_END, 'experiences')

# TODO switch below
env_controller = PepperEnvController()
# env_controller = HumanEnvController('test2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = env_controller.get_action_and_state_size()

def compute_returns(next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + settings.gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def save_fig(labels):
  for label in labels:
    result_data_controller.plot(label)
    result_data_controller.plot(label)
    result_data_controller.plot(label)

def load_results_and_experiences(file_name_end):
  if os.path.exists('../training_results/results-{}.csv'.format(file_name_end)):
    cumulative_rewards = result_data_controller.get_data('cumulative_reward')
    succeeds = result_data_controller.get_data('succeed')
    actor_losses = result_data_controller.get_data('actor_loss')
    critic_losses = result_data_controller.get_data('critic_loss')
  else:
    cumulative_rewards = []
    succeeds = []
    actor_losses = []
    critic_losses = []

  if os.path.exists('../training_results/experiences-{}.csv'.format(file_name_end)):
    states = experience_controller.get_data('state')
    actions = experience_controller.get_data('action')
    rewards = experience_controller.get_data('reward')
    next_states = experience_controller.get_data('next_state')
  else:
    states = []
    actions = []
    rewards = []
    next_states = [] 
  return cumulative_rewards, succeeds, actor_losses, critic_losses, states, actions, rewards, next_states

def select_action(dist, epsilon):
  # probs = tensor([5.0393e-03, 3.0475e-01,... 2.6321e-01], device='cuda:0', grad_fn=<SoftmaxBackward0>)
  if random.random() < epsilon:
    print('random is chosen')
    action = torch.tensor(random.randint(0, action_size-1)).to(device='cuda:0')
  else:
    print('maximum is chosen')
    action = dist.sample()
  return action

def trainIters(actor, critic, file_name_end):
  # Initialize the result data
  cumulative_rewards, succeeds, actor_losses, critic_losses, \
    states, actions, rewards, next_states = load_results_and_experiences(file_name_end)
  test_rewards = []

  for nepisode in range(len(cumulative_rewards), settings.nepisodes):
    print("Episode: " + str(nepisode))
    cumulative_rewards.append(0)
    log_probs = []
    values = []
    entropy = 0
    tensor_rewards = []
    masks = []

    state = env_controller.get_state()
    print("State\n", state)
    if nepisode < settings.nepisodes - 1:
      epsilon = settings.epsilon_begin + (settings.epsilon_end - settings.epsilon_begin) * nepisode / settings.nepisodes
    else:
      epsilon = settings.epsilon_end
    print("Epsilon", epsilon)

    for i in range(NSTEP):
      optimizerA.zero_grad()
      optimizerC.zero_grad()
      state = torch.FloatTensor(state).to(device)
      dist, value = actor(state), critic(state)
      action = select_action(dist, epsilon)  # TODO NEP
      # action, log_prob = human_data_controller.get_action(i) # TODO csv

      env_controller.publish_action(action)  # TODO NEP
      next_state, reward, done = env_controller.step(i)
      print('Next_state\n', next_state)
      print('Reward', reward)
      print('Done', done)
  
      log_prob = dist.log_prob(action).unsqueeze(0)
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      tensor_rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
      masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
      cumulative_rewards[-1] += reward
      test_rewards.append(reward)
      states.append(state.cpu().tolist())
      action_num = action.cpu().numpy()
      actions.append(action_num)
      rewards.append(reward)
      next_states.append(next_state)

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
      ## Save the experiences
      experience_controller.write('experiences', state=states, action=actions, reward=rewards, next_state=next_states)   
    # Save the model and optimizer by episodes
    # TODO train
    torch.save(actor.state_dict(), 'model/actor.pkl')
    torch.save(critic.state_dict(), 'model/critic.pkl')
    torch.save(optimizerA.state_dict(), 'optimizer/optimizerA.pkl')
    torch.save(optimizerC.state_dict(), 'optimizer/optimizerC.pkl')

    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)

    returns = compute_returns(next_value, tensor_rewards, masks)
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

    # TODO when you run train and test, switch the below two lines
    ## Save the results
    result_data_controller.write('results', cumulative_reward=cumulative_rewards, succeed=succeeds, actor_loss=actor_losses, critic_loss=critic_losses)
    # result_data_controller.write('results', cumulative_reward=cumulative_rewards, succeed=succeeds, actor_loss=actor_losses, critic_loss=critic_losses)


    # Optimize the weight parameters
    actor_loss.backward()  # calculate gradient
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

  torch.save(actor.state_dict(), 'model/actor.pkl')
  torch.save(critic.state_dict(), 'model/critic.pkl')
  torch.save(optimizerA.state_dict(), 'optimizer/optimizerA.pkl')
  torch.save(optimizerC.state_dict(), 'optimizer/optimizerC.pkl')

  # TODO when you run train and test, switch the below two lines
  save_fig(['cumulative_reward', 'actor_loss', 'critic_loss'])
  save_fig(['cumulative_reward'])

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
  trainIters(actor, critic, FILE_NAME_END)
  # trainIters(actor, critic, 'test1')
