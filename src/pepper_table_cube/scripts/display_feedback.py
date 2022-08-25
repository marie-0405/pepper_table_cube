#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'feedback_ros'
FILE_NAME_REAL = 'feedback_real'
HUMAN_DATA = 'pushing_task6'

RATE = 25.0
JOINT_DIC = {"RElbowRoll": "右肘ロール角",
                "RElbowYaw": "右肘ヨー角",
                "RShoulderPitch": "右肩ピッチ角",
                "RShoulderRoll": "右肩ロール角"}

class Feedback:
  def __init__(self, joint_names=None, file_name=FILE_NAME):
    self.df = pd.read_csv('~/catkin_ws/src/research_pepper/pepper_table_cube/scripts/test/data/{}.csv'.format(FILE_NAME))
    # self.df_real = pd.read_csv('~/catkin_ws/src/research_pepper/pepper_table_cube/scripts/test/data/{}.csv'.format(FILE_NAME_REAL))

    if joint_names:
      self.joint_dic = {}
      for joint_name in joint_names:
        self.joint_dic[joint_name] = JOINT_DIC[joint_name]
    else:
      self.joint_dic = JOINT_DIC
    
    self.colors=["blue", "red", "green", "orange"]
  
  def display(self, english=True, font_size=25):
    plt.figure(figsize=(15, 8))
    # plt.subplot(1, 2, 1)

    ## display human data
    # human_df = pd.read_csv('../../human/joint_data/{}_3d.csv'.format(HUMAN_DATA))

    # human_stride = float(1.0 / 4.0)
    # human_df = human_df.drop(human_df.shape[0] - 1)
    # human_time = np.linspace(0, human_stride * human_df.shape[0], human_df.shape[0])

    ## Settings of graph
    stride = float(1 / RATE)
    time = np.linspace(0, stride * self.df.shape[0], self.df.shape[0])
    plt.xlim(0, stride * self.df.shape[0])
    plt.ylim(-0.8, 0.25)
    stride = float(1.0/15.0)
    # time_real = np.linspace(0, stride * self.df_real.shape[0], self.df_real.shape[0])
    
    plt.rcParams["font.size"] = 20

    ## Plot data
    if english:
      for i, joint in enumerate(self.joint_dic):
        plt.plot(time, self.df['desired_' + joint], '-', c=self.colors[i], label="human")
      for i, joint in enumerate(self.joint_dic):
        plt.plot(time, self.df['actual_' + joint], '--', c=self.colors[i], label="ROS")
        # plt.plot(time, self.df['actual_' + joint], '--', c=self.colors[i], label="Real")
        # plt.plot(human_time, human_df[joint_name], '--', c=self.colors[i])


      plt.xlabel('Time [s]', fontsize=font_size)
      plt.ylabel('Angle[rad]', fontsize=font_size) 

    else:
      for i, joint in enumerate(self.joint_dic):
        plt.plot(time, self.df['desired_' + joint], c='red', label="人間の" + self.joint_dic[joint])
      for i, joint in enumerate(self.joint_dic):
        plt.plot(time, self.df['actual_' + joint], '--', c='blue', label="ROS上の" + self.joint_dic[joint])
        # plt.plot(time_real, self.df_real['actual_' + joint], '--', c='green', label="リアルの" + self.joint_dic[joint])
      
      # for i, joint in enumerate(self.joint_dic):
      #   plt.plot(time_real, self.df_real['actual_' + joint], '--', c=self.colors[i], label="実際の" + self.joint_dic[joint])

      plt.xlabel('時間 [s]', fontsize=font_size)
      plt.ylabel('関節角度[rad]', fontsize=font_size)
    
    # plt.legend(bbox_to_anchor=(1.05, 0.5, 1.0, 0.5), loc="upper left")
    plt.legend()
    plt.show()


if __name__ == '__main__':
  joint_names = ["RShoulderRoll"]
  feedback = Feedback(joint_names)
  feedback.display(english=True)



  


  

  



 