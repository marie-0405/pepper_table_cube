#!/usr/bin/env python
# coding: UTF-8

import math
import numpy as np
import pandas as pd
import torch

import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HumanDataController:
  def __init__(self, file_name):
    # read human data from csv file
    self.df = pd.read_csv('./human_data/{}_distance.csv'.format(file_name))
  
  def get_action(self, index):
    """
    stateとnext_stateを比較して、最も大きく変化がある関節をactionとする
    """
    joint_names = ['RShoulderRoll', 'RShoulderPitch', 'RElbowRoll', 'RElbowYaw']
    state = np.array(self.get_data(joint_names, index=index-1))
    next_state = np.array(self.get_data(joint_names, index=index))
    print('state', state)
    print('next_state', next_state)
    diff = abs(next_state - state)
    print('diff', diff)
    max_i = np.argmax(diff)
    # print(max_i)
    prob = diff[max_i] / np.sum(diff)
    # print(prob)
    log_prob = math.log(prob)
    if next_state[max_i] - state[max_i] >= 0:
      print("Increment {}: {}".format(joint_names[max_i], 2*max_i))
      action = 2 * max_i
    else:
      print("Decrement {}: {}".format(joint_names[max_i], 2*max_i + 1))
      action = 2*max_i + 1
    action = torch.tensor(action, dtype=torch.int32, device=device)
    print('action', action)
    return action, log_prob
  
  def get_data(self, column_names, index=0):
    arr = []
    for column_name in column_names:
      arr.append(self.df.iloc[index][column_name])
    return arr
  
  def calculate_reward_done(self, distance1, distance2):
    weight1 = settings.w1
    weight2 = settings.w2
    base_reward = settings.base_reward
    reward = base_reward - (distance1 * weight1) - distance2 * weight2
    done = bool(distance2 <= 5)
    return reward, done
  
