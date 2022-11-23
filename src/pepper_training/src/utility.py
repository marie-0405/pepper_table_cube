import os
import random
import torch

import settings

def compute_returns(next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + settings.gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def save_fig(result_data_controller, labels):
  for label in labels:
    result_data_controller.plot(label)

def load_results_and_experiences(result_data_controller, experience_controller, file_name_end):
  if os.path.exists('../training_results/{}-{}/results-{}.csv'.format(settings.date, file_name_end, file_name_end)):
    cumulative_rewards = result_data_controller.get_data('cumulative_reward')
    succeeds = result_data_controller.get_data('succeed')
    actor_losses = result_data_controller.get_data('actor_loss')
    critic_losses = result_data_controller.get_data('critic_loss')
  else:
    cumulative_rewards = []
    succeeds = []
    actor_losses = []
    critic_losses = []

  if os.path.exists('../training_results/{}-{}/experiences-{}.csv'.format(settings.date, file_name_end, file_name_end)):
    states = experience_controller.get_data('state')
    actions = experience_controller.get_data('action')
    rewards = experience_controller.get_data('reward')
    next_states = experience_controller.get_data('next_state')
    dists = experience_controller.get_data('distribution')
  else:
    states = []
    actions = []
    rewards = []
    next_states = []
    dists = []
  return cumulative_rewards, succeeds, actor_losses, critic_losses, states, actions, rewards, next_states, dists

def select_action(dist, epsilon, action_size):
  # probs = tensor([5.0393e-03, 3.0475e-01,... 2.6321e-01], device='cuda:0', grad_fn=<SoftmaxBackward0>)
  if random.random() < epsilon:
    print('random is chosen')
    action = torch.tensor(random.randint(0, action_size-1)).to(device='cuda:0')
  else:
    print('maximum is chosen')
    action = dist.sample()
  return action

def compute_MSE_joint_reward(robot_states, human_states):
  return mean_squared_error(robot_states, human_states)