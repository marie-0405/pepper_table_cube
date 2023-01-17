# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import rospkg
from scipy import stats
import sys
import settings

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22

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
      'success_probability': '{}/training_results/{}-{}/success-{}.png'.format(pkg_path, settings.date, file_name_end, file_name_end),
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

  def plot(self, label, ylim=[-35, 170]):
    result_df = self._read()
    plt.figure()
    result_df[label].plot(figsize=(11, 7), label=label.capitalize().replace('_', ' '))
    average = self.get_average(label)
    # plt.plot(np.arange(0, len(result_df)), 
    #          np.full(len(result_df), average),
    #          label="Average= {}".format(average))

    # Axis label
    plt.xlabel("The number of episode")
    plt.ylabel(label.capitalize().replace('_', ' '))
    # plt.ylabel("Reward")  # TODO test

    plt.ylim(ylim)
    # plt.legend(edgecolor="black")
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

  def plot_success_probability(self):
    """
    10回ごとの成功確率を表示する
    """
    div_num = 25
    result_df = self._read()
    success_probabilities = []
    for i in range(int(len(result_df) / div_num)):
      tf_list = result_df["succeed"][div_num*i:div_num*i + div_num].to_list()
      success_probabilities.append(tf_list.count(True) / div_num)
    
    plt.figure(figsize=[11, 6])
    x_linspace = np.linspace(0, len(result_df), int(len(result_df) / div_num))
    print(x_linspace)
    plt.plot(x_linspace, success_probabilities)
    
    # Axis label
    plt.xlabel("The number of episodes")
    plt.ylabel("Probability of success")

    plt.ylim([0, 1.2])
    plt.savefig(self.file_path["success_probability"])
  
def plot_success(with_human, without_human):
  """
  10回ごとの成功確率を表示する
  """
  div_num = 25
  with_human_df = ResultController(with_human)._read()
  without_human_df = ResultController(without_human)._read()
  
  success_probabilities_with_human = []
  average_with_human = with_human_df["succeed"].to_list().count(True) / len(with_human_df["succeed"])
  for i in range(int(len(with_human_df) / div_num)):
    tf_list = with_human_df["succeed"][div_num*i:div_num*i + div_num].to_list()
    success_probabilities_with_human.append(tf_list.count(True) / div_num)
  cp_success_probabilities_with_human = success_probabilities_with_human.copy()
  for v in cp_success_probabilities_with_human:
    if v == 0.0:
      cp_success_probabilities_with_human.remove(v)
  average_with_human = sum(cp_success_probabilities_with_human[-10:])/len(cp_success_probabilities_with_human[-10:])

  success_probabilities_without_human = []
  average_without_human = without_human_df["succeed"].to_list().count(True) / len(without_human_df["succeed"])
  for i in range(int(len(without_human_df) / div_num)):
    tf_list = without_human_df["succeed"][div_num*i:div_num*i + div_num].to_list()
    success_probabilities_without_human.append(tf_list.count(True) / div_num)
  cp_success_probabilities_without_human = success_probabilities_without_human.copy()
  for v in cp_success_probabilities_without_human:
    if v == 0.0:
      cp_success_probabilities_without_human.remove(v)
  average_without_human = sum(cp_success_probabilities_without_human[-12:])/len(cp_success_probabilities_without_human[-12:])

  
  plt.figure(figsize=[10, 7])
  x_linspace = np.linspace(0, len(with_human_df), int(len(with_human_df) / div_num))
  x_linspace2 = np.linspace(div_num * 8, len(with_human_df), int(len(with_human_df) / div_num))
  # plt.plot(x_linspace, success_probabilities_with_human, label="With human data", color="tab:orange")
  # plt.plot(x_linspace, success_probabilities_without_human, label="Without human data", color="tab:blue")
  # plt.plot(x_linspace, [average_with_human for i in range(len(x_linspace))], linestyle="dashed", color="tab:orange", label="Average with human")
  # plt.plot(x_linspace, [average_without_human for i in range(len(x_linspace))], linestyle="dashed", color="tab:blue", label="Average without human")

  plt.plot(x_linspace, success_probabilities_with_human, label="With human data", color="tab:orange")
  plt.plot(x_linspace, success_probabilities_without_human, label="Without human data", color="tab:blue")
  plt.plot(x_linspace2, [average_with_human for i in range(len(x_linspace))], linestyle="dashed", color="tab:orange", label="Average with human data")
  plt.plot(x_linspace2, [average_without_human for i in range(len(x_linspace))], linestyle="dashed", color="tab:blue", label="Average without human data")


  # ans = stats.ttest_ind(cp_success_probabilities_without_human[-6:], cp_success_probabilities_with_human[-6:], alternative="less")
  ans = stats.ttest_ind(cp_success_probabilities_without_human[-12:], cp_success_probabilities_with_human[-12:], alternative="less")
  print(ans)
  print(cp_success_probabilities_with_human[3:])
  print(cp_success_probabilities_without_human[1:])
  print(len(cp_success_probabilities_with_human))
  
  # Axis label
  plt.xlabel("The number of episodes")
  plt.ylabel("Probability of success")

  plt.ylim([-0.2, 1.2])
  # plt.savefig(self.file_path["success_probability"])
  plt.legend(edgecolor="black")
  plt.show()

def plot_success_network(small, medium, large):
  """
  10回ごとの成功確率を表示する
  """
  div_num = 25
  small_df = ResultController(small)._read()
  medium_df = ResultController(medium)._read()
  large_df = ResultController(large)._read()
  
  success_probabilities_small = []
  for i in range(int(len(small_df) / div_num)):
    tf_list = small_df["succeed"][div_num*i:div_num*i + div_num].to_list()
    success_probabilities_small.append(tf_list.count(True) / div_num)

  success_probabilities_medium = []
  for i in range(int(len(medium_df) / div_num)):
    tf_list = medium_df["succeed"][div_num*i:div_num*i + div_num].to_list()
    success_probabilities_medium.append(tf_list.count(True) / div_num)

  success_probabilities_large = []
  for i in range(int(len(large_df) / div_num)):
    tf_list = large_df["succeed"][div_num*i:div_num*i + div_num].to_list()
    success_probabilities_large.append(tf_list.count(True) / div_num)
  
  plt.figure(figsize=[10, 7])
  x_linspace = np.linspace(0, len(small_df), int(len(small_df) / div_num))

  plt.plot(x_linspace, success_probabilities_small, label="Small size", color="tab:orange")
  plt.plot(x_linspace, success_probabilities_medium, label="Medium size", color="tab:blue")
  plt.plot(x_linspace, success_probabilities_large, label="Large size", color="tab:green")

  # Axis label
  plt.xlabel("The number of episodes")
  plt.ylabel("Probability of success")

  plt.ylim([-0.2, 1.2])
  plt.legend(edgecolor="black")
  plt.show()
  
    
if __name__ == '__main__':
  file_name_end = sys.argv[1] if len(sys.argv)==2 else ''
  # file_name_end = ['test1', 'test2', 'test3']\
  # file_name_end = settings.file_name_end
  # result_controller = ResultController("random_cube_with_human_1000")
  # result_controller.plot("cumulative_reward", [-35, 175])
  # result_controller = ResultController("joints_and_vectors_500")
  # result_controller.plot_success_probability()
  # result_controller.plot('cumulative_reward', [-35, 175])
  # plot_success("little_random_cube_500", "little_random_cube_with_human_500")
  # plot_success("random_cube_with_human", "random_cube") # Ttest_indResult(statistic=-2.8951195919207797, pvalue=0.004823261034250219)  
  # plot_success("joints_and_vectors_500", "epsilon_greedy_500")
  plot_success_network("small_size_network", "medium_size_network", "large_size_network")
  # plot_success("with_human_500", "joints_and_vectors_500") # Ttest_indResult(statistic=-1.832641132962928, pvalue=0.04021317315874035) 
  # plot_success("negative_reward_500", "joints_and_vectors_500")
  # for fne in file_name_end:
  #   experience_controller = ResultController('positive_epsilon_off', 'experiences')
  #   experience_controller.plot_arrays('distribution')
