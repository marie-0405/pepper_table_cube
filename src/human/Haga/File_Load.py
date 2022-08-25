import os
import pandas as pd
import numpy as np

cwd = os.getcwd()
path_data = cwd + "\\Shoulder_left.trc"
df = pd.read_table(path_data, sep="\t", header=3, skiprows=[4], usecols=range(0,176)) #skiprowsおよびusecolsはlistで指定
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
