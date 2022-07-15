import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from Classes.Plot import Plot
from Classes.Euler import Euler
from Classes.Rotation import Rotation
from Classes.Body_R import Body_R


def load_markers_data(file_path):

    marker_data = (np.load(file_path, allow_pickle='TRUE')).item() #.item()で初めて元の辞書となる

    return marker_data

def extract_frame(marker_data, frame):

    marker_data_frame = {}

    for marker in marker_data.keys():
        marker_data_frame[marker] = marker_data[marker][:, frame]

    return marker_data_frame

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
marker_data_frame = extract_frame(marker_data=marker_data, frame=3000)

fig, ax = Plot.plot_frame(marker_data_frame)



#以後セグメントのトランスフォーム
Ot, Xt, Yt, Zt, Rt = Body_R.R_trunk(SJN=marker_data_frame["SJN"], SXS=marker_data_frame["SXS"], CV7=marker_data_frame["CV7"], TV7=marker_data_frame["TV7"])
Orf, Xrf, Yrf, Zrf, Rrf = Body_R.R_forearm(USP=marker_data_frame["RUSP"], HLE=marker_data_frame["RHLE"], HME=marker_data_frame["RHME"], RSP=marker_data_frame["RRSP"], way="right")
Orh2, Xrh2, Yrh2, Zrh2, Rrh2 = Body_R.R_humerus2(CAJ=marker_data_frame["RCAJ"], HLE=marker_data_frame["RHLE"], HME=marker_data_frame["RHME"], Yt=Yt, Yf=Yrf)
Olf, Xlf, Ylf, Zlf, Rlf = Body_R.R_forearm(USP=marker_data_frame["LUSP"], HLE=marker_data_frame["LHLE"], HME=marker_data_frame["LHME"], RSP=marker_data_frame["LRSP"], way="left")
Olh2, Xlh2, Ylh2, Zlh2, Rlh2 = Body_R.R_humerus2(CAJ=marker_data_frame["LCAJ"], HLE=marker_data_frame["LHLE"], HME=marker_data_frame["LHME"], Yt=Yt, Yf=Ylf)
Op, Xp, Yp, Zp, Rp = Body_R.R_pelvis(RIAS=marker_data_frame["RIAS"], LIAS=marker_data_frame["LIAS"], RIPS=marker_data_frame["RIPS"], LIPS=marker_data_frame["LIPS"])

print("Rt = {0}\n".format(Rt))
print("Rrh2 = {0}\n".format(Rrh2))
print("Rrf = {0}\n".format(Rrf))
print("Rlh2 = {0}\n".format(Rlh2))
print("Rlf = {0}\n".format(Rlf))
print("Rp = {0}\n".format(Rp))

Plot.draw_R(ax, Ot, Xt, Yt, Zt)
Plot.draw_R(ax, Orh2, Xrh2, Yrh2, Zrh2)
Plot.draw_R(ax, Orf, Xrf, Yrf, Zrf)
Plot.draw_R(ax, Olh2, Xlh2, Ylh2, Zlh2)
Plot.draw_R(ax, Olf, Xlf, Ylf, Zlf)
Plot.draw_R(ax, Op, Xp, Yp, Zp)
plt.show()


Rs = Rotation.relative_R(parent=Rt, child=Rrh2) # JCS of shoulder
Re = Rotation.relative_R(parent=Rrh2, child=Rrf) # JCS of elbow
Rpt = Rotation.relative_R(parent=Rp, child=Rt) # the trunk RS relative to pelvis RS

print("Rs = {0}\n".format(Rs))
print("Re = {0}\n".format(Re))
print("Rpt = {0}\n".format(Rpt))

s_adduction, s_flexion, s_forward_flexion = Rotation.to_euler(Rs, "XYZ", False)
e_adduction, e_pronation, e_flexion = Rotation.to_euler(Re, "XYZ", False)
pt_lateral_bending, pt_rotation, pt_extension = Rotation.to_euler(Rpt, "XYZ", False)
p_alpha, p_beta, p_gamma = Rotation.to_euler(Rp, "XYZ", False)

print("shoulder:  adduction = {0}°  flexion = {1}°  forward flexion = {2}°".format(np.degrees(s_adduction), np.degrees(s_flexion), np.degrees(s_forward_flexion)))
print("elbow:  adduction = {0}°  pronation = {1}°  flexion = {2}°".format(np.degrees(e_adduction),np.degrees(e_pronation), np.degrees(e_flexion)))
print("pt:  lateral_bending = {0}°  rotation = {1}°  extension = {2}°".format(np.degrees(pt_lateral_bending),np.degrees(pt_rotation), np.degrees(pt_extension)))
print("pelvis:  alpha = {0}°\n         beta = {1}°\n         gamma = {2}°\n".format(np.degrees(p_alpha),np.degrees(p_beta), np.degrees(p_gamma)))
print("pt:      alpha = {0}°\n         beta = {1}°\n         gamma = {2}°\n".format(np.degrees(pt_lateral_bending),np.degrees(pt_rotation), np.degrees(pt_extension)))

seq = "XYZ"

rs = Rot.from_matrix(matrix=Rs)
re = Rot.from_matrix(matrix=Re)
eulers_scipy_s = rs.as_euler(seq=seq, degrees=True)
eulers_scipy_e = re.as_euler(seq=seq, degrees=True)

s_alpha, s_beta, s_gamma = Rotation.to_euler(Rs, seq, False)
e_alpha, e_beta, e_gamma = Rotation.to_euler(Re, seq, False)

print("shoulder:")
print("scipy:  alpha = {0}°  beta = {1}°  gamma = {2}°".format(eulers_scipy_s[0], eulers_scipy_s[1], eulers_scipy_s[2]))
print("mine:  alpha = {0}°  beta = {1}°  gamma = {2}°\n".format(np.degrees(s_alpha), np.degrees(s_beta), np.degrees(s_gamma)))
print("elbow:")
print("scipy:  alpha = {0}°  beta = {1}°  gamma = {2}°".format(eulers_scipy_e[0], eulers_scipy_e[1], eulers_scipy_e[2]))
print("mine:  alpha = {0}°  beta = {1}°  gamma = {2}°\n".format(np.degrees(e_alpha), np.degrees(e_beta), np.degrees(e_gamma)))


#test
s_eulers = Rotation.to_euler(Rs, seq, to_degree=True)

Rs_Rot = Rot.from_euler(seq, [s_alpha, s_beta, s_gamma], degrees=False)
Rs_Rot = Rs_Rot.as_matrix()
Rs_mine = Euler.to_rotation_matrix(s_eulers, seq, degrees=True)

print("Rs_Rot = {0}".format(Rs_Rot))
print("Rs_mine = {0}\n".format(Rs_mine))
