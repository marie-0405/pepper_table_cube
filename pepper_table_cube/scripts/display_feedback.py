#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'feedback'
HUMAN_DATA = 'pushing_task4'
RATE = 25.0

if __name__ == '__main__':
  plt.figure(figsize=(15, 8))
  # plt.rcParams["font.family"] = "Times New Roman"
  # plt.rcParams["font.family"] = 'MS ゴシック'

  plt.rcParams["font.size"] = 16

  df = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME))
  human_df = pd.read_csv('../../human/joint_data/{}_3d.csv'.format(HUMAN_DATA))

  stride = float(1 / RATE)
  time = np.linspace(0, stride * df.shape[0], df.shape[0])
  human_stride = float(1.0 / 2.3)
  human_df = human_df.drop(human_df.shape[0] - 1)
  human_time = np.linspace(0, human_stride * human_df.shape[0], human_df.shape[0])

  plt.subplot(1, 2, 1)
  # plt.plot(time, df['actual_RElbowRoll'], '-', c='red')
  # plt.plot(time, df['actual_RElbowYaw'], '-', c='blue')
  # plt.plot(time, df['actual_RShoulderPitch'], '-', c='orange')
  # plt.plot(time, df['actual_RShoulderRoll'], '-', c='black')
  # plt.plot(time, df['desired_RElbowRoll'], '--', c='red')
  # plt.plot(time, df['desired_RElbowYaw'], '--', c='blue')
  # plt.plot(time, df['desired_RShoulderPitch'], '--', c='orange')
  # plt.plot(time, df['desired_RShoulderRoll'], '--', c='black')

  plt.plot(time, df['actual_RElbowRoll'], '-', c='red', label='実際の右肘ロール角')
  plt.plot(time, df['actual_RElbowYaw'], '-', c='blue', label='実際の右肘ヨー角')
  plt.plot(time, df['actual_RShoulderPitch'], '-', c='orange', label='実際の右肩ピッチ角')
  plt.plot(time, df['actual_RShoulderRoll'], '-', c='black', label='実際の右肩ロール角')
  plt.plot(time, df['desired_RElbowRoll'], '--', c='red', label='目標の右肘ロール角')
  plt.plot(time, df['desired_RElbowYaw'], '--', c='blue', label='目標の右肘ヨー角')
  plt.plot(time, df['desired_RShoulderPitch'], '--', c='orange', label='目標の右肩ピッチ角')
  plt.plot(time, df['desired_RShoulderRoll'], '--', c='black', label='目標の右肩ロール角')

  # plt.xlabel('Time [s]')
  # plt.ylabel('Angle[rad]')  
  plt.xlabel('時間 [s]')
  plt.ylabel('関節角度[rad]')
  plt.xlim(0, stride * df.shape[0])
  plt.legend(bbox_to_anchor=(1.05, 0.5, 1.0, 0.5), loc="upper left")
  # plt.tight_layout()
  plt.show()