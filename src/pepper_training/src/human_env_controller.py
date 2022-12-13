#!/usr/bin/env python
# coding: UTF-8

import math
import numpy as np
import pandas as pd
import torch
import IPython

import settings
from interface.ienvironment_controller import IEnvController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HumanEnvController(IEnvController):
  def __init__(self, file_name_end, nvideo, state_names, joint_names):
    # read human data from csv file
    super().__init__()
    self.df = pd.read_csv('./human_data/{}-{}/distance_{}.csv'.format(settings.video_date, file_name_end, nvideo))
    print("Before", len(self.df))
    self.df = self.trim()
    print("Trimmed", len(self.df))
    self.step_size = len(self.df)
    self.state_names = state_names
    self.joint_names = joint_names
    
  def get_action_and_state_size(self):
    # TODO change if you change state or action
    action_size = 10
    state_size = 11
    print('action_size', action_size)
    print('state_size', state_size)
    return action_size, state_size
  
  def trim(self):
    """
    The beginning of the video shows human not moving.
    So we trim the beginning for getting ideal movement of human.
    """
    # Don't use 1th row data because it is not correct
    trimmed_df = self.df[1:]
    # Find 1st line number sarisfied with Distance1[i-1] - Distance1[i] > 3
    Distance1 = trimmed_df['Distance1']
    begin_num = len(trimmed_df)
    for i in range(len(Distance1) - 1):
      print(Distance1.iloc[i], Distance1.iloc[i+1])
      if abs(Distance1.iloc[i] - Distance1.iloc[i+1]) > 3:
        begin_num = i
        break
    trimmed_df = trimmed_df[begin_num:]
    return trimmed_df
  
  def get_action(self, index):
    """
    stateとnext_stateを比較して、最も大きく変化がある関節をactionとする
    """
    state = np.array(self._get_data_from_file(self.joint_names, index=index-1))
    next_state = np.array(self._get_data_from_file(self.joint_names, index=index))
    print('state', state)
    print('next_state', next_state)
    diff = abs(next_state - state)
    print('diff', diff)
    max_i = np.argmax(diff)
    # log_prob = math.log(prob)
    if next_state[max_i] - state[max_i] >= 0:
      print("Increment {}: {}".format(self.joint_names[max_i], 2*max_i))
      action = 2 * max_i
    else:
      print("Decrement {}: {}".format(self.joint_names[max_i], 2*max_i + 1))
      action = 2*max_i + 1
    action = torch.tensor(action, dtype=torch.int32, device=device)
    print('action', action)
    return action
  
  def _get_data_from_file(self, column_names, index=0):
    list = []
    for column_name in column_names:
      list.append(self.df.iloc[index][column_name])
    return list
  
  def get_state(self, index=0) -> list:
    state = self._get_data_from_file(self.state_names, index)
    print('state', state)
    return state

  def _get_reward(self, index):
    weight1 = settings.w1
    weight2 = settings.w2
    before_distances = self._get_data_from_file(['Distance1', 'Distance2'], index-1)
    distances = self._get_data_from_file(['Distance1', 'Distance2'], index)
    r1 = self._calculate_reward(weight1, before_distances[0], distances[0])
    r2 = self._calculate_reward(weight2, before_distances[1], distances[1])
    reward = r1 + r2
    
    # additional reward
    if distances[0] < 55:  # TODO if hand reaches the cube
      reward += 1
    
    return reward
  
  def _calculate_reward(self, weight, before_distance, distance):
    if before_distance + 2.5 > distance:
      reward = 1 * weight
    else:
      reward = -1 * weight
    return reward
  
  def _is_task_ok(self, index):
    distances = self._get_data_from_file(['Distance1', 'Distance2'], index)
    if distances[1] < 10:
      return True
    return False

  def _get_done(self, distance2):
    done = bool(distance2 <= 0.03)
    return done
  
  def step(self, index):
    next_state = self.get_state(index)
    reward = self._get_reward(index)
    done = self._is_task_ok(index)
    if done:
      reward = 100
    return next_state, reward, done
    
  
