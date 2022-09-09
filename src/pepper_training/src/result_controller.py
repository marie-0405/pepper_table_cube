# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import rospkg
import sys

from result import Result

class ResultController():

  def __init__(self, file_name_end):
    # Set the file path ファイルパスを設定
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('pepper_training')
    
    # Windows用
    pkg_path = os.getcwd()
    pkg_path = pkg_path[:-4]

    self.file_name_end = file_name_end
    self.file_path = {
      'results': pkg_path + '/training_results/results-'+ file_name_end + '.csv',
      'reward': pkg_path + '/training_results/reward-'+ file_name_end + '.png',
      'q_matrix': pkg_path + '/training_results/q_matrix-'+ file_name_end + '.txt',
      'experience': pkg_path + '/training_results/experience-'+ file_name_end + '.csv',
      'actor_loss': pkg_path + '/training_results/actor-loss-'+ file_name_end + '.png',
      'critic_loss': pkg_path + '/training_results/critic-loss-'+ file_name_end + '.png',
    }

  def write(self, rewards, succeeds, experiences='', q_matrix='', actor_losses='', critic_losses=''):
    if not actor_losses:
      actor_losses = [0 for _ in range(len(rewards))]
    if not critic_losses:
      actor_losses = [0 for _ in range(len(rewards))]

    result = Result(rewards, succeeds, experiences, q_matrix, actor_losses, critic_losses)

    result.df.to_csv(self.file_path['results'])
    result.experiences.to_csv(self.file_path['experience'])
    with open(self.file_path['q_matrix'], 'w') as f:
      f.write(str(result.q_matrix))
  
  def _read(self):
    result_df = pd.read_csv(self.file_path['results'], engine="python")
    return result_df

  def count_q_matrix(self):
    with open(self.file_path['q_matrix']) as f:
      lines = f.read()
      return lines.count(':')
  
  def get_average(self, label):
    result_df = self._read()
    average = result_df[label].mean()
    return average

  def plot(self, label):
    result_df = self._read()
    plt.figure()
    result_df[label].plot(figsize=(11, 6), label=label.capitalize())
    average = self.get_average(label)
    plt.plot(np.arange(0, len(result_df)), 
             np.full(len(result_df), average),
             label="Average= {}".format(average))

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))

    plt.ylim([-25.0, 15.0])
    plt.legend()
    plt.savefig(self.file_path[label])

if __name__ == '__main__':
  file_name_end = sys.argv[1] if len(sys.argv)==2 else ''
  result_controller = ResultController(file_name_end)
  result_controller.plot('reward')
