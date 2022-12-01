from importlib.metadata import distribution
import IPython
import json
from turtle import st
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import time

import torch
import torch.optim as optim

from actor import Actor
from critic import Critic
from result_controller import ResultController
from human_env_controller import HumanEnvController
from pepper_env_controller import PepperEnvController
from utility import load_results_and_experiences, select_action, save_fig, compute_returns
import settings

NSTEP = settings.nsteps

# TODO switch below
pepper_env_controller = PepperEnvController()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
state_names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw", 'Vector1X', 'Vector1Y', 'Vector1Z', 'Vector2X', 'Vector2Y', 'Vector2Z']

def trainIters(actor, critic, file_name_end):
  # Initialize the result data
  cumulative_rewards, succeeds, actor_losses, critic_losses, average_rewards, \
    states, actions, rewards, next_states, dists = load_results_and_experiences(result_data_controller, experience_controller, file_name_end)
  test_rewards = []
  for nepisode in range(int(len(cumulative_rewards) / settings.nvideos), int(settings.nepisodes / settings.nvideos)):
    print("nepipsodes", nepisode)
    print("cumulative", len(cumulative_rewards))
    print("begin nvideo", len(cumulative_rewards) % settings.nvideos)
    for nvideo in range(len(cumulative_rewards) % settings.nvideos, settings.nvideos):
      print("Episode: " + str(nepisode))
      print("Video:" + str(nvideo))
      cumulative_rewards.append(0)
      log_probs = []
      values = []
      entropy = 0
      tensor_rewards = []
      masks = []
      
      # print(file_name_end)
      human_env_controller = HumanEnvController(settings.video_file_name, nepisode, state_names, joint_names)
      state = pepper_env_controller.get_state()
      human_state = human_env_controller.get_state()
      # print("Pepper joint state", state[:5])
      # print("State\n", state)
      if nepisode < settings.nepisodes - 1:
        epsilon = settings.epsilon_begin + (settings.epsilon_end - settings.epsilon_begin) * nepisode / settings.nepisodes
      else:
        epsilon = settings.epsilon_end
      # print("Epsilon", epsilon)
      
      ## Send step_size to Pepper environment
      # time.sleep(4)
      pepper_env_controller.publish_step_size(human_env_controller.step_size)
      for i in range(human_env_controller.step_size):
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        state = torch.FloatTensor(state).to(device)
        dist, value = actor(state), critic(state)
        # action = select_action(dist, epsilon, action_size)  # TODO NEP
        action = dist.sample()  # TODO epsilon-off
        # print('Distribution', dist.probs)
        # action, log_prob = human_data_controller.get_action(i) # TODO csv

        ## TODO Additional reward that is  "- MSE(human_joint_state, pepper_joint_state)"
        pepper_joint_state = state[:5]
        human_joint_state = human_state[:5]
        
        mse_human_pepper_joint = mean_squared_error(pepper_joint_state.cpu().tolist(), human_joint_state)
        print("Human joint reward", -mse_human_pepper_joint)
        pepper_env_controller.publish_action(action)  # TODO NEP

        next_state, reward, done = pepper_env_controller.step(i)
        print('Next_state\n', next_state)
        print('Reward', reward)
        print('Done', done)
        reward -= mse_human_pepper_joint # TODO pepper human
    
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
        dists.append(dist.probs.cpu().tolist())

        # Judgement for End or Not
        if done:
          succeeds.append(True)
          print('Iteration: {}, Score: {}'.format(nepisode, i))
          break
        else:
          if i == human_env_controller.step_size - 1:
            succeeds.append(False)
          else:
            state = next_state
            
        ## Save the experiences
        experience_controller.write('experiences', state=states, action=actions, reward=rewards, next_state=next_states, distribution=dists)   
      # Save the model and optimizer by episodes
      # TODO test
      torch.save(actor.state_dict(), actor_path)
      torch.save(critic.state_dict(), critic_path)
      torch.save(optimizerA.state_dict(), optimizerA_path)
      torch.save(optimizerC.state_dict(), optimizerC_path)

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
      ## TODO Additional loss value of defference between human joint and pepper joint
      # critic_loss = advantage.pow(2).mean() + mse_human_pepper_joint


      # Save the losses for plot
      actor_losses.append(actor_loss.detach().item())
      critic_losses.append(critic_loss.detach().item())
      
      print("When average_reward step_size", human_env_controller.step_size)
      average_rewards.append(cumulative_rewards[-1] / human_env_controller.step_size)
      print(average_rewards)

      # IPython.embed()
      ## Save the results 
      result_data_controller.write('results', cumulative_reward=cumulative_rewards, succeed=succeeds, actor_loss=actor_losses, critic_loss=critic_losses, average_reward=average_rewards)

      # TODO test 
      # Optimize the weight parameters
      actor_loss.backward()  # calculate gradient
      critic_loss.backward()
      optimizerA.step()
      optimizerC.step()

      # TODO test
      save_fig(result_data_controller, ['cumulative_reward', 'actor_loss', 'critic_loss'])
      # save_fig(result_data_controller, ['cumulative_reward'])
      
      experience_controller.plot_arrays('distribution')

  ## TODO test
  torch.save(actor.state_dict(), actor_path)
  torch.save(critic.state_dict(), critic_path)
  torch.save(optimizerA.state_dict(), optimizerA_path)
  torch.save(optimizerC.state_dict(), optimizerC_path)

  

if __name__ == '__main__':
  action_size, state_size = pepper_env_controller.get_action_and_state_size()
  for file_name_end in settings.file_name_end:
    # TODO test comment out while training
    test_file_name_end = 'test_' + file_name_end
    print(file_name_end)
    
    training_results_dir = '../training_results/{}-{}/'.format(settings.date, file_name_end)
    actor_path = '{}model/actor-{}.pkl'.format(training_results_dir, file_name_end)
    critic_path = '{}model/critic-{}.pkl'.format(training_results_dir, file_name_end)
    optimizerA_path = '{}optimizer/optimizerA-{}.pkl'.format(training_results_dir, file_name_end)
    optimizerC_path = '{}optimizer/optimizerC-{}.pkl'.format(training_results_dir, file_name_end)
    
    # TODO test
    result_data_controller = ResultController(file_name_end)
    experience_controller = ResultController(file_name_end, 'experiences')
    # result_data_controller = ResultController(test_file_name_end)
    # experience_controller = ResultController(test_file_name_end, 'experiences')

    if os.path.exists(actor_path):
      actor = Actor(state_size, action_size, 256, 512).to(device)
      actor.load_state_dict(torch.load(actor_path))
      actor.train()
      # actor.eval()  # TODO test
      print('Actor Model loaded')

      optimizerA = optim.Adam(actor.parameters(), lr=settings.lr)
      optimizerA.load_state_dict(torch.load(optimizerA_path))
      print('Actor Optimizer loaded')
    else:
      actor = Actor(state_size, action_size, 256, 512).to(device)
      optimizerA = optim.Adam(actor.parameters(), lr=settings.lr)
    if os.path.exists(critic_path):
      critic = Critic(state_size, action_size, 256, 512).to(device)
      critic.load_state_dict(torch.load(critic_path))
      critic.train()
      # critic.eval()  # TODO test
      print('Critic Model loaded')

      optimizerC = optim.Adam(critic.parameters(), lr=settings.lr)
      optimizerC.load_state_dict(torch.load(optimizerC_path))
      print('Critic Optimizer loaded')
    else:
      critic = Critic(state_size, action_size, 256, 512).to(device)
      optimizerC = optim.Adam(critic.parameters(), lr=settings.lr)
    trainIters(actor, critic, file_name_end)
    # trainIters(actor, critic, test_file_name_end)
