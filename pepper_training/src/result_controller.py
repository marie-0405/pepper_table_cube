# coding: UTF-8
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rospkg
import sys

from result import Result

class ResultController():

  def __init__(self, file_name_end):
    # Set the file path ファイルパスを設定
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pepper_training')
    self.file_name_end = file_name_end
    self.reward_fig = pkg_path + '/training_results/reward-'+ file_name_end + '.png'
    self.reward_file_path = pkg_path + '/training_results/reward-'+ file_name_end + '.csv'
    self.q_matrix_file_path = pkg_path + '/training_results/q_matrix-'+ file_name_end + '.txt'

  def write(self, rewards, succeeds, q_matrix):
    """Output result dataframe to csv"""
    result = Result(rewards, succeeds, q_matrix)
    result.df.to_csv(self.reward_file_path)
    with open(self.q_matrix_file_path, "w") as f:
      f.write(str(result.q_matrix))
  
  def _read(self):
    result_df = pd.read_csv(self.reward_file_path, engine="python")
    return result_df

  def count_q_matrix(self):
    q_matrix = {}
    with open(self.q_matrix_file_path) as f:
      lines = f.read()
      return lines.count(':')
  
  def plot_reward(self):
    result_df = self._read()
    plt.figure()
    result_df["reward"].plot(figsize=(11, 6), label="Reward")

    # Plot the average
    average = result_df["reward"].mean()
    plt.plot(np.arange(0, len(result_df)), 
             np.full(len(result_df), average),
             label="Average= {}".format(average))

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel("Reward")

    plt.ylim([-25.0, 15.0])
    plt.legend()
    plt.savefig(self.reward_fig)

if __name__ == '__main__':
  file_name_end = sys.argv[1] if len(sys.argv)==2 else ''
  result_controller = ResultController(file_name_end)
  result_controller.plot_reward()
