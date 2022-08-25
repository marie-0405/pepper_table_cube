#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'data'
RATE = 25.0

if __name__ == '__main__':
  plt.figure()
  # plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["font.size"] = 16

  df = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME))

  stride = float(1 / RATE)
  time = np.linspace(0, stride * df.shape[0], df.shape[0])

  # plt.plot(time, df['hand_x'], '-', c='red')
  # plt.plot(time, df['hand_y'], '-', c='blue')
  # plt.plot(time, df['target_x'], '--', c='red')
  # plt.plot(time, df['target_y'], '--', c='blue')
  # plt.plot(time, df['cube_x'], ':', c='red')
  # plt.plot(time, df['cube_y'], ':', c='blue')

  plt.plot(time, df['hand_x'], '-', c='red', label='手先x')
  plt.plot(time, df['hand_y'], '-', c='blue', label='手先y')
  plt.plot(time, df['target_x'], '--', c='red', label='目標x')
  plt.plot(time, df['target_y'], '--', c='blue', label='目標y')
  plt.plot(time, df['cube_x'], ':', c='red', label='キューブx')
  plt.plot(time, df['cube_y'], ':', c='blue', label='キューブy')

  # plt.xlabel('Time [s]')
  # plt.ylabel('Position[m]')
  plt.xlabel('時間 [s]')
  plt.ylabel('位置[m]')
  plt.xlim(0, stride * df.shape[0] * 1.6)
  plt.legend()
  plt.show()