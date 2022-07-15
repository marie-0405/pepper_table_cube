import os
import pandas as pd
import numpy as np
import rapidjson
from Classes.Euler import Euler
from Classes.Rotation import Rotation
from Classes.Body_R import Body_R


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

marker_data = load_markers_data(file_path)
marker_data = change_data_R(marker_data=marker_data, R=R)

n_frame = marker_data["CV7"].shape[1]  # Your total number of frame (shape[1]:the number of column)
segments = ['RightArm', 'LeftArm', 'RightForearm', 'LeftForearm']
parameters = ['O', 'R']
dataR_for_json = []

for i in range(n_frame):
    dict = {}

    Ot, Xt, Yt, Zt, Rt = Body_R.R_trunk(SJN=marker_data["SJN"][:, i], SXS=marker_data["SXS"][:, i],
                                        CV7=marker_data["CV7"][:, i], TV7=marker_data["TV7"][:, i])
    Orf, Xrf, Yrf, Zrf, Rrf = Body_R.R_forearm(USP=marker_data["RUSP"][:, i], HLE=marker_data["RHLE"][:, i],
                                               HME=marker_data["RHME"][:, i], RSP=marker_data["RRSP"][:, i],
                                               way="right")
    Orh2, Xrh2, Yrh2, Zrh2, Rrh2 = Body_R.R_humerus2(CAJ=marker_data["RCAJ"][:, i], HLE=marker_data["RHLE"][:, i],
                                                     HME=marker_data["RHME"][:, i], Yt=Yt, Yf=Yrf)
    Olf, Xlf, Ylf, Zlf, Rlf = Body_R.R_forearm(USP=marker_data["LUSP"][:, i], HLE=marker_data["LHLE"][:, i],
                                               HME=marker_data["LHME"][:, i], RSP=marker_data["LRSP"][:, i], way="left")
    Olh2, Xlh2, Ylh2, Zlh2, Rlh2 = Body_R.R_humerus2(CAJ=marker_data["LCAJ"][:, i], HLE=marker_data["LHLE"][:, i],
                                                     HME=marker_data["LHME"][:, i], Yt=Yt, Yf=Ylf)
    
    dict['RightArm'] = {}
    dict['LeftArm'] = {}
    dict['RightForearm'] = {}
    dict['LeftForearm'] = {}
    dict['RightArm']['O'] = Orh2.tolist()
    dict['LeftArm']['O'] = Olh2.tolist()
    dict['RightForearm']['O'] = Orf.tolist()
    dict['LeftForearm']['O'] = Olf.tolist()
    dict['RightArm']['R'] = Rrh2.reshape(-1).tolist()
    dict['LeftArm']['R'] = Rlh2.reshape(-1).tolist()
    dict['RightForearm']['R'] = Rrf.reshape(-1).tolist()
    dict['LeftForearm']['R'] = Rlf.reshape(-1).tolist()

    dataR_for_json.append(dict)


json_dataR = rapidjson.dumps(dataR_for_json)
filename = "\\Shoulder_left_R.txt"
folder = cwd + "\\Data"
with open(folder + filename, 'w') as f:
    f.write(json_dataR)