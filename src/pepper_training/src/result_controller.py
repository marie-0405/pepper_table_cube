# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rospkg
import sys

class ResultController():

  def __init__(self, file_name_end, file_name='results'):
    # TODO Set the file path ファイルパスを設定
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pepper_training')
    
    # TODO Windows用
    # pkg_path = os.getcwd()
    # pkg_path = pkg_path[:-4]

    self.file_name_end = file_name_end
    self.file_name = file_name
    self.file_path = {
      'results': pkg_path + '/training_results/results-'+ file_name_end + '.csv',
      'cumulative_reward': pkg_path + '/training_results/reward-'+ file_name_end + '.png',
      'q_matrix': pkg_path + '/training_results/q_matrix-'+ file_name_end + '.txt',
      'experiences': pkg_path + '/training_results/experiences-'+ file_name_end + '.csv',
      'actor_loss': pkg_path + '/training_results/actor-loss-'+ file_name_end + '.png',
      'critic_loss': pkg_path + '/training_results/critic-loss-'+ file_name_end + '.png',
    }

  def write(self, file_name='results', **kwargs):
    self.df = pd.DataFrame(kwargs)
    self.df.to_csv(self.file_path[file_name])
    
  def _read(self):
    result_df = pd.read_csv(self.file_path[self.file_name], engine="python")
    return result_df
  
  def get_data(self, column):
    result_df = self._read()
    return result_df[column].to_numpy().tolist()

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
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.figure()
    result_df[label].plot(figsize=(11, 6), label=label.capitalize())
    average = self.get_average(label)
    plt.plot(np.arange(0, len(result_df)), 
             np.full(len(result_df), average),
             label="Average= {}".format(average))

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))
    # plt.ylabel("Reward")  # TODO test

    plt.ylim([-30.0, 0.0])
    plt.legend(edgecolor="black")
    plt.savefig(self.file_path[label])

if __name__ == '__main__':
  file_name_end = sys.argv[1] if len(sys.argv)==2 else ''
  # file_name_end = ['test1', 'test2', 'test3']
  file_name_end = ['decrease_the_sizes_reward_positive']
  for fne in file_name_end:
    result_controller = ResultController(fne)
    result_controller.plot('cumulative_reward')
