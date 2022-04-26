# coding: UTF-8
import matplotlib.pyplot as plt
import pandas as pd
import rospkg
import rospy

class Information(object):

  def __init__(self, file_name, **kwargs):
    # Set the file path ファイルパスを設定
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pepper_training')
    self.file_path = pkg_path + '/training_results/' + file_name

    if kwargs:
      self.df = pd.DataFrame(kwargs)
  
  def write(self):
    # output dataframe to csv
    self.df.to_csv(self.file_path)

  def read(self):
    self.df = pd.read_csv(self.file_path)

  def plot(self, column):
    """
    param
      - column:string column name you want to plot
    """
    self.read()
    self.df[column].plot(figsize=(11, 6))
    plt.show()


