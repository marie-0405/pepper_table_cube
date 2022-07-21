# coding: UTF-8
import pandas as pd
import os

def main():
  cwd = os.getcwd()
  data = pd.read_csv(cwd + '/mocap_data/RightToLeft.csv')
  # print(data)

  df = pd.read_table(cwd + '/mocap_data/RightToLeft.csv', sep="\t", header=3, skiprows=[4], usecols=range(0,176))
  print(df)

  markers = df.columns.values
  print(markers)

  data = df.values.T #行と列のラベルを外したndarrayの転置
  dict = {}

  for i in range(0,58):
      dict[markers[i*3+2]] = data[i*3+2:(i+1)*3+2, :]
  print(dict)

  path_save = cwd + "\\Shoulder_left_data.npy"
  np.save(path_save, dict)


if __name__ == '__main__':
  main()