#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FILE_NAME = 'feedback'
RATE = 25.0

if __name__ == '__main__':

  df_desired = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME)).filter(like='desired', axis=1)[1:]
  df_actual = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME)).filter(like='actual', axis=1)[1:]
  df_error = pd.read_csv('./test/data/{}_error.csv'.format(FILE_NAME))[1:]  # remove initial data because including zero

  error_rate = (abs(df_error).values / abs(df_desired).values) * 100
  print(np.mean(error_rate, axis=0))
