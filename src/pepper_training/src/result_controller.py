# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import rospkg
import sys
import settings

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

class ResultController():

  def __init__(self, file_name_end, file_name='results'):
    # TODO Set the file path ファイルパスを設定
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('pepper_training')
    
    # TODO Windows用
    pkg_path = os.getcwd()
    pkg_path = pkg_path[:-4]

    self.file_name_end = file_name_end
    self.file_name = file_name
    self.file_path = {
      'results': '{}/training_results/{}-{}/results-{}.csv'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'cumulative_reward': '{}/training_results/{}-{}/reward-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'q_matrix': '{}/training_results/{}-{}/q_matrix-{}.txt'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'experiences': '{}/training_results/{}-{}/experiences-{}.csv'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'actor_loss': '{}/training_results/{}-{}/actor-loss-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'critic_loss': '{}/training_results/{}-{}/critic-loss-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'distribution': '{}/training_results/{}-{}/distribution-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
      'average_reward': '{}/training_results/{}-{}/avg_reward-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
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

  def plot(self, label, ylim=[-35, 150]):
    result_df = self._read()
    plt.figure()
    result_df[label].plot(figsize=(11, 6), label=label.capitalize().replace('_', ' '))
    average = self.get_average(label)
    plt.plot(np.arange(0, len(result_df)), 
             np.full(len(result_df), average),
             label="Average= {}".format(average))

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))
    # plt.ylabel("Reward")  # TODO test

    plt.ylim(ylim)
    plt.legend(edgecolor="black")
    plt.savefig(self.file_path[label])

  def plot_average_reward(self, label, ylim=[-35, 25]):
    result_df = self._read()
    plt.figure()
    result_df['cumulative_reward']

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))
    # plt.ylabel("Reward")  # TODO test

    plt.ylim(ylim)
    plt.legend(edgecolor="black")
    plt.savefig()
  
  def plot_batch(self, label, num_batch):
    result_df = self._read()
    plt.figure(figsize=(11, 6))
    # result_df[label].plot(figsize=(11, 6), label=label.capitalize().replace('_', ' '))
    # average = self.get_average(label)
    batch_result = []
    for i in range(int(len(result_df[label]) / num_batch)):
      start_index = i * num_batch
      end_index = i * num_batch + num_batch
      batch_result.append(result_df[label][start_index:end_index].mean())
    print(batch_result)
    plt.plot([i for i in range(int(len(result_df[label]) / num_batch))], batch_result, label=label.capitalize().replace('_', ' '))
    
    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))
    # plt.ylabel("Reward")  # TODO test

    plt.ylim([-30.0, -15.0])
    plt.show()
    # plt.savefig(self.file_path[label])
  
  def plot_arrays(self, label):
    """
    plot the data of array
    """
    result_df = self._read()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(11, 6))
    
    df = result_df[label].str.strip('[]')
    df = df.str.split(pat=',', n=-1, expand=True)
    df = df.astype(float)
    
    for i in range(df.shape[1]):
      df[i].astype(float).plot(label=i)

    # Axis label
    plt.xlabel("The number of steps")
    plt.ylabel(label.capitalize().replace('_', ' '))
    
    plt.ylim([0.0, 1.0])
    plt.legend(edgecolor="black")
    plt.savefig(self.file_path[label])

if __name__ == '__main__':
  file_name_end = sys.argv[1] if len(sys.argv)==2 else ''
  # file_name_end = ['test1', 'test2', 'test3']
  file_name_end = ['positive_epsilon_off']
  result_controller = ResultController('test_positive_epsilon_off')
  result_controller.plot('cumulative_reward', [-60, 170])
  # for fne in file_name_end:
  #   experience_controller = ResultController('positive_epsilon_off', 'experiences')
  #   experience_controller.plot_arrays('distribution')
