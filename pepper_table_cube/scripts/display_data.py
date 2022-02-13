#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'r_hand'
RATE = 25.0

if __name__ == '__main__':
  plt.figure()
  df = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME))
  stride = float(1 / RATE)
  time = np.linspace(0, stride * df.shape[0], df.shape[0])
  plt.plot(time, df['x'])
  plt.plot(time, df['y'])
  plt.plot(time, df['z'])
  plt.xlabel('Time [s]')
  plt.ylabel('Position[m]')
  plt.show()