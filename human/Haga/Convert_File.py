import os
import pandas as pd
import numpy as np
import rapidjson


def load_markers_data(file_path):

    marker_data = (np.load(file_path, allow_pickle='TRUE')).item() #.item()で初めて元の辞書となる

    return marker_data

def change_data_R(marker_data, R):

    marker_data_new_R = {}

    for marker in marker_data.keys():
        marker_data_new_R[marker] = np.dot(R, marker_data[marker])

    return marker_data_new_R



cwd = os.getcwd()
file_path = cwd + "\\Shoulder_left_data.npy"

R = np.array([[0,0,-1],[-1,0,0],[0,1,0]])

marker_data = load_markers_data(file_path)  # Your dictionary of 3D joint positions
marker_data = change_data_R(marker_data=marker_data, R=R)

n_frame = marker_data["CV7"].shape[1]  # Your total number of frame (shape[1]:the number of column)
joints = marker_data.keys()
data_for_json = {}

for joint in joints:
    data_for_json[joint] = []
    for i in range(n_frame):
        d = {'x': marker_data[joint][0, i], 'y': marker_data[joint][1, i], 'z': marker_data[joint][2, i]}
        data_for_json[joint].append(d)

json_data = rapidjson.dumps(data_for_json)
filename = "\\Shoulder_left.txt"
folder = cwd + "\\Data"
with open(folder + filename, 'w') as f:
    f.write(json_data)