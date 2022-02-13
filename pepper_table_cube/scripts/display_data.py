#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'data'
RATE = 25.0

if __name__ == '__main__':
  plt.figure()
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["font.size"] = 16

  df = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME))

  stride = float(1 / RATE)
  time = np.linspace(0, stride * df.shape[0], df.shape[0])

  plt.plot(time, df['hand_x'], '-', c='red')
  plt.plot(time, df['hand_y'], '-', c='blue')
  plt.plot(time, df['target_x'], '--', c='red')
  plt.plot(time, df['target_y'], '--', c='blue')
  plt.plot(time, df['cube_x'], ':', c='red')
  plt.plot(time, df['cube_y'], ':', c='blue')

  plt.xlabel('Time [s]')
  plt.ylabel('Position[m]')
  plt.xlim(0, stride * df.shape[0] * 1.5)
  plt.legend()
  plt.show()