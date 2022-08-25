# coding: UTF-8
import matplotlib.pyplot as plt
import pandas as pd
import rospkg
import rospy
import pandas


class Information(object):

  def __init__(self, file_name, **kwargs):
    # Set the file path ファイルパスを設定
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pepper_training')
    self.reward_file_path = pkg_path + '/training_results/' + file_name
    self.matrix_file_path = pkg_path + '/training_results/q_matrix.txt'

    if kwargs:
      self.df = pd.DataFrame(kwargs)
  
  def write(self, q_matrix):
    # output dataframe to csv
    self.df.to_csv(self.reward_file_path)
    with open(self.matrix_file_path, "w") as f:
      f.write(str(q_matrix))

  def read(self):
    self.df = pd.read_csv(self.reward_file_path, engine="python")

  def plot(self, column):
    """
    param
      - column:string column name you want to plot
    """
    self.read()
    plt.figure()
    self.df[column].plot(figsize=(11, 6))
    plt.xlabel("The number of episode")
    plt.ylabel("Reward")
    plt.show()


