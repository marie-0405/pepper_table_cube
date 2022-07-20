from Classes.IMU_Attitude import Quaternion
from Classes.Info import Data_info
from Classes.Data import Data_IMU_offline
from Classes.Calibration import Calibration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from Classes.IMU_Attitude import Convertion_angles_rotation_matrix
from Right_Arm_OMC import CreateIKdata
from scipy.signal import butter, filtfilt

class Filtering():
    def __init__(self):
        pass

    def butter_function(self, cutoff, fs, order=5):
        nyq = 0.5 * fs #ナイキスト周波数
        normal_cutoff = cutoff / nyq #正規化周波数
        b, a = butter(order, normal_cutoff)
        return b, a

    def butter_lowpass_function(self, data, cutoff, fs, order=5):
        b, a = self.butter_function(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def butterworth_filter(self, data, cutoff, fs, order):
        data_x = np.array(data.iloc[:,0])
        data_y = np.array(data.iloc[:,1])
        data_z = np.array(data.iloc[:,2])

        data_x = self.butter_lowpass_function(data_x, cutoff, fs, order)
        data_y = self.butter_lowpass_function(data_y, cutoff, fs, order)
        data_z = self.butter_lowpass_function(data_z, cutoff, fs, order)

        data_x = pd.DataFrame(data_x)
        data_y = pd.DataFrame(data_y)
        data_z = pd.DataFrame(data_z)

        data = pd.concat([data_x, data_y, data_z], axis=1, ignore_index=True)

        return data

    def shimple_moving_average_filter(self, num, data):
        """
        Shimple moving average filter

        Parameters :
            num - int:
            data - DataFrame (dim: ,3):

        Returns:
            DataFrame (dim: ,3):
        """
        supplement = data.iloc[0:num - 1, :]
        supplement.columns = [0, 1, 2]

        kernel = np.ones(num) / num
        data = np.array(data)
        data_filtered_0 = np.convolve(data[:,0], kernel, mode='valid')
        data_filtered_1 = np.convolve(data[:,1], kernel, mode='valid')
        data_filtered_2 = np.convolve(data[:,2], kernel, mode='valid')
        data_filtered = pd.DataFrame([data_filtered_0,data_filtered_1,data_filtered_2]).T

        data_filtered = pd.concat([supplement, data_filtered], ignore_index=True)

        return data_filtered

    def exponential_moving_average_filter(self, data, a):
        """
        Exponential moving average filter (Low-pass filter)
         - extract the gravitational acceleration from the output of Accelerometer

        Parameters :
            data - DataFrame (dim: ,3):
            a - float: coefficient

        Returns:
            DataFrame (dim: ,3):
        """

        data_filtered = pd.DataFrame(np.empty((data.shape[0], 3)))
        data_filtered.iloc[0, 0] = data.iloc[0, 0]
        data_filtered.iloc[0, 1] = data.iloc[0, 1]
        data_filtered.iloc[0, 2] = data.iloc[0, 2]
        for i in range(1, data.shape[0]):
            data_filtered.iloc[i, 0] = a * data.iloc[i, 0] + (1 - a) * data_filtered.iloc[i - 1, 0]
            data_filtered.iloc[i, 1] = a * data.iloc[i, 1] + (1 - a) * data_filtered.iloc[i - 1, 1]
            data_filtered.iloc[i, 2] = a * data.iloc[i, 2] + (1 - a) * data_filtered.iloc[i - 1, 2]

        return data_filtered
def get_the_calibrated_output(shimmer_name, data_info, calibration):
    if shimmer_name == 'A833':
        number = 0
    else:
        number = 1

    data = Data_IMU_offline()
    data.set_obj(data_info=data_info, calibration=calibration)

    path_file = 'Data\\Shimmer\\Elbow_Motion_Tracking\\Tetsuya_Abe\\IMU_cut_spline\\Flexion_Extension_opt.npy'
    data_uncal = np.load(path_file)
    data_uncal = data_uncal[:, :, number]

    data_uncal = pd.DataFrame(data_uncal)

    a_uncal = data_uncal.iloc[:, [0, 1, 2]]
    w_uncal = data_uncal.iloc[:, [3, 4, 5]]
    m_uncal = data_uncal.iloc[:, [6, 7, 8]]

    # data = Data_IMU_offline()
    # data.set_obj(data_info=data_info, calibration=calibration)
    # path = 'C:/Users/abetetsuya/PycharmProjects/IMU_Calibration_Server_Python/Data/Shimmer/Elbow_Motion_Tracking/Tetsuya_Abe/IMU/Flexion_Extension/Tetsuya_Abe_4_' + shimmer_name + '.csv'
    # data.set_path_file(path)
    # data.get_data()
    # data_uncal = pd.DataFrame(data.df)
    #
    # a_uncal = data_uncal.loc[:,['XAccel', 'YAccel', 'ZAccel']]
    # w_uncal = data_uncal.loc[:,['XGyro', 'YGyro', 'ZGyro']]
    # m_uncal = data_uncal.loc[:,['XMag', 'YMag', 'ZMag']]

    calibration.get(display=False)
    #  --------  Accelerometer calibration [m/sec^2] --------
    temp1_a = np.dot(np.linalg.inv(data.calibration.IMU_calibration_data[shimmer_name]['Ra']), np.linalg.inv(data.calibration.IMU_calibration_data[shimmer_name]['Ka']))
    temp2_a = a_uncal.T - data.calibration.IMU_calibration_data[shimmer_name]['ba']
    a_cal = np.dot(temp1_a, temp2_a)
    a_cal = pd.DataFrame(a_cal.T)

    #  --------  Gyroscope calibration [rad/sec] --------
    temp1_w = np.dot(np.linalg.inv(data.calibration.IMU_calibration_data[shimmer_name]['Rg']), np.linalg.inv(data.calibration.IMU_calibration_data[shimmer_name]['Kg']))
    temp2_w = w_uncal.T - data.calibration.IMU_calibration_data[shimmer_name]['bg']
    w_cal = np.dot(temp1_w, temp2_w)
    w_cal = pd.DataFrame(w_cal.T)
    for i in range(w_cal.shape[0]):
        w_cal.iloc[i,:]= Convertion_angles_rotation_matrix.deg2rad(w_cal.iloc[i,:])

    #  --------  Magnetometer calibration [-] --------
    temp1_m = np.dot(np.linalg.inv(-data.calibration.IMU_calibration_data[shimmer_name]['Rm']), np.linalg.inv(data.calibration.IMU_calibration_data[shimmer_name]['Km']))
    temp2_m = m_uncal.T - data.calibration.IMU_calibration_data[shimmer_name]['bm']
    m_cal = np.dot(temp1_m, temp2_m)
    m_cal = pd.DataFrame(m_cal.T)

    #  --------  Filtering of Accelerometer & Magnetometer --------
    filtering = Filtering()

    # -------- SMA filter --------
    a_cal = filtering.shimple_moving_average_filter(num=10, data=a_cal)
    m_cal = filtering.shimple_moving_average_filter(num=10, data=m_cal)

    # -------- EMG filter --------
    # a_cal = filtering.exponential_moving_average_filter(a_cal, a=0.04)
    # m_cal = filtering.exponential_moving_average_filter(m_cal, a=0.04)

    # -------- Butterworth filter --------
    # a_cal = filtering.butterworth_filter(a_cal, cutoff=2, fs=frequency, order=5)
    # m_cal = filtering.butterworth_filter(m_cal, cutoff=2, fs=frequency, order=5)

    return a_cal, w_cal, m_cal
class Extended_kalman_filter:

    def __init__(self, dt, sequence):

        self.dt = dt
        self.sequence = sequence

        # Initial states
        q = np.array([[1], [0], [0], [0]])     # Quaternion
        b = np.array([[0], [0], [0]])          # Bias

        self.accelReference = np.array([[0], [0], [1]])
        self.magReference = np.array([[0], [0.653], [-0.758]])

        # self.mag_Ainv = np.array([[ 2.06423128e-03, -1.04778851e-04, -1.09416190e-06],
        #                           [-1.04778851e-04,  1.91693168e-03,  1.79409312e-05],
        #                           [-1.09416190e-06,  1.79409312e-05,  1.99819154e-03]])
        # self.mag_b = np.array([[80.51340236], [37.08931099], [105.6731885]])

        self.X = np.concatenate([q, b], axis=0)  # Self.X

        self.A, self.B = self.define_A_B(q=q, dt=self.dt)

        # # W (noise in the predicted state)
        # self.W = np.zeros(shape=(7, 1))

        #self.yHatBar = np.zeros(shape=(1, 3))

        # Predicted process covariance matrix
        self.P = np.identity(7) * 0.01

        # Process variance - Error in the process of the covariance matrix
        self.Q = np.identity(7) * 0.01

        # Observation error
        self.R = np.identity(6) * 0.01

        # Conversion matrix
        #self.H = self.define_H(q=q)

        # Simple identity matrix
        self.I = np.eye(7)

    def set_obj(self, quaternion):

        self.quaternion = quaternion
        self.quaternion_list = [self.X[0:4]]

        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X[0:4])
        self.X_list = [euler_angles.squeeze().tolist()]

    def define_A_B(self, q, dt):

        # A and B used in (X_p = A.X + B.U + W) to compute states matrix
        Sq = np.array([[-q[1], -q[2], -q[3]],
                       [q[0], -q[3], q[2]],
                       [q[3], q[0], -q[1]],
                       [-q[2], q[1], q[0]]]).squeeze()

        tmp1 = np.concatenate([np.identity(4), -dt / 2 * Sq], axis=1)
        tmp2 = np.concatenate([np.zeros((3, 4)), np.identity(3)], axis=1)

        A = np.concatenate([tmp1, tmp2], axis=0)
        B = np.concatenate([dt / 2 * Sq, np.zeros((3, 3))], axis=0)

        return A, B

    def normalize_quaternion(self, q):

        mag = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5

        return q / mag

    def getAccelVector(self, a):

        return a / np.linalg.norm(a)

    def getMagVector(self, m):

        # magGaussRaw = np.dot(self.mag_Ainv, np.array(m) - self.mag_b)
        # magGauss_N = np.dot(self.quaternion.convert_q_to_R(q=self.X[0:4]), magGaussRaw)

        magGauss_N = np.dot(self.quaternion.convert_q_to_R(q=self.X_p[0:4]), m) #Body frame →　World frame
        magGauss_N[2] = 0
        magGauss_N = magGauss_N / (magGauss_N[0] ** 2 + magGauss_N[1] ** 2) ** 0.5
        magGuass_B = np.dot(self.quaternion.convert_q_to_R(q=self.X_p[0:4]).transpose(), magGauss_N)  #World frame → Body frame

        return magGuass_B

    def define_J(self, q, reference):

        # Jacobian matrix

        e00 = q[0] * reference[0] + q[3] * reference[1] - q[2] * reference[2]
        e01 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        e02 = -q[2] * reference[0] + q[1] * reference[1] - q[0] * reference[2]
        e03 = -q[3] * reference[0] + q[0] * reference[1] + q[1] * reference[2]
        e10 = -q[3] * reference[0] + q[0] * reference[1] + q[1] * reference[2]
        e11 = q[2] * reference[0] - q[1] * reference[1] + q[0] * reference[2]
        e12 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        e13 = -q[0] * reference[0] - q[3] * reference[1] + q[2] * reference[2]
        e20 = q[2] * reference[0] - q[1] * reference[1] + q[0] * reference[2]
        e21 = q[3] * reference[0] - q[0] * reference[1] - q[1] * reference[2]
        e22 = q[0] * reference[0] + q[3] * reference[1] - q[2] * reference[2]
        e23 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        J = 2 * np.array([[e00, e01, e02, e03],
                          [e10, e11, e12, e13],
                          [e20, e21, e22, e23]]).squeeze()
        return J

    def define_H(self, q):

        hPrime_a = self.define_J(q=q, reference=self.accelReference)
        hPrime_m = self.define_J(q=q, reference=self.magReference)

        tmp1 = np.concatenate((hPrime_a, np.zeros((3, 3))), axis=1)
        tmp2 = np.concatenate((hPrime_m, np.zeros((3, 3))), axis=1)

        H = np.concatenate((tmp1, tmp2), axis=0)

        return H

    # def predict_acc_mag(self, q):
    #     return np.dot(self.define_H(q), self.X_p)

    def predict_acc_mag(self, q):

        # Accel and mag
        accelBar = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=q).transpose(), self.accelReference)
        magBar = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=q).transpose(), self.magReference)

        return np.concatenate((accelBar, magBar), axis=0)

    def update(self, U, a, m, dt):

        self.U = np.array(U)[:, np.newaxis]

        a = np.array(a)[:, np.newaxis]
        m = np.array(m)[:, np.newaxis]

        q = self.X[0:4]

        # Predicted states matrix (X_p = A.X + B.U + W)
        self.A, self.B = self.define_A_B(q=q, dt=dt)
        self.X_p = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        self.X_p[0:4] = self.normalize_quaternion(self.X_p[0:4])

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Q

        # Conversion matrix
        self.H = self.define_H(q=self.X_p[0:4])

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        self.K = np.dot(np.dot(self.P_p, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        self.yHatBar = self.predict_acc_mag(q=self.X_p[0:4])

        # Current observations matrix (Y = C.Y + Z)
        #magGuass_B = self.getMagVector(m)
        magGuass_B = m / np.linalg.norm(m)
        accel_B = self.getAccelVector(a)
        Y = np.concatenate((accel_B, magGuass_B), axis=0)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(self.K, Y - self.yHatBar)
        self.X[0:4] = self.normalize_quaternion(self.X[0:4])

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(self.K, self.H), self.P_p)

        # Information to keep
        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X[0:4])

        self.quaternion_list.append(self.X[0:4])
        self.X_list.append(euler_angles.squeeze().tolist())
class Adaptive_complementary_Kalman_Filter_Sub():
    def __init__(self, dt):

        # Initial state
        self.X = np.array([[0], [0]])
        self.P = np.identity(2)
        # Process variance - Error in the process of the covariance matrix
        self.Q = np.array([[0.3, 0.3], [0.3, 0.3]])
        # Observation error
        self.R = 0.01
        self.F = np.array([[1, dt], [0, 1]])
        self.B = np.array([[dt], [0]])
        # Conversion matrix
        self.H = np.array([1, 0]).reshape([1,2])

    def update(self, u, z):

        # Predicted states matrix (X_p = F.X + B.u)
        self.X_p = np.dot(self.F, self.X) + np.dot(self.B, u)

        # Predicted covariance matrix (P_p = F.P.F' + Q)
        self.P_p = np.dot(np.dot(self.F, self.P), self.F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        self.K = np.multiply(np.dot(self.P_p, self.H.reshape(-1, 1)), 1 / (np.dot(np.dot(self.H, self.P_p), self.H.reshape(-1, 1)) + self.R))

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.multiply(self.K, (z - np.dot(self.H, self.X_p)))

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = self.P_p - np.dot(np.dot(self.K, self.H), self.P_p)
class Adaptive_Complementary_Kalman_Filter():

    def __init__(self, dt, sequence):
        self.sequence = sequence
        self.roll_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)
        self.pitch_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)
        self.yaw_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)

        self.X_list = []
        self.X_list.append([0,0,0])

    def set_obj(self, quaternion):
        self.quaternion = quaternion

    def get_roll(self, a):
        roll = math.atan2(a[1], math.sqrt(a[0] ** 2.0 + a[2] ** 2.0))
        return roll

    def get_pitch(self, a):
        pitch = math.atan2(-a[0], math.sqrt(a[1] ** 2.0 + a[2] ** 2.0))
        return pitch

    def get_yaw(self, m, roll, pitch):
        h_x = m[1] * math.cos(roll) - m[2] * math.sin(roll)
        h_y = m[0] * math.cos(pitch) + m[1] * math.sin(pitch) * math.sin(roll) + m[2] * math.sin(pitch) * math.cos(roll)
        yaw = math.atan2(h_y, h_x)
        return yaw

    def main(self, w, a, m):

        roll = self.get_roll(a)
        self.roll_kf.update(u=w[0], z=roll)
        roll = self.roll_kf.X[0,0]

        pitch = self.get_pitch(a)
        self.pitch_kf.update(u=w[1], z=pitch)
        pitch = self.pitch_kf.X[0,0]

        yaw = self.get_yaw(m, roll, pitch)
        self.yaw_kf.update(u=w[2], z=yaw)
        yaw = self.yaw_kf.X[0,0]

        roll_angle = Convertion_angles_rotation_matrix.rad2deg(roll)
        pitch_angle = Convertion_angles_rotation_matrix.rad2deg(pitch)
        yaw_angle = Convertion_angles_rotation_matrix.rad2deg(yaw)

        self.X_list.append([roll_angle, pitch_angle, yaw_angle])
class Gradient_Descent_Kalman_Filter():

    def __init__(self, sequence):
        # Initial values
        self.sequence = sequence
        self.X = np.array([[1], [0], [0], [0]])
        self.P = np.identity(4) * 2

        # the process covariance matrix
        # var = np.array([(0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2])
        # Q1 = [var[0] + var[1] + var[2], -var[0] + var[1] - var[2], -var[0] - var[1] + var[2], var[0] - var[2] - var[2]]
        # Q2 = [-var[0] + var[1] - var[2], var[0] + var[1] + var[2], var[0] - var[1] - var[2], -var[0] - var[2] + var[2]]
        # Q3 = [-var[0] - var[1] + var[2], var[0] - var[1] - var[2], var[0] + var[1] + var[2], -var[0] + var[2] - var[2]]
        # Q4 = [var[0] - var[1] - var[2], -var[0] - var[1] + var[2], -var[0] + var[1] - var[2], var[0] + var[2] + var[2]]
        # self.Q = np.array([[Q1], [Q2], [Q3], [Q4]])

        self.Q = np.identity(4) * 0.01

        # the transformation matrix (obsevation　→　state)
        self.H = np.identity(4)

        # the identity matrix
        self.I = np.identity(4)

        # the observations covariance matrix
        self.R = np.identity(4) * 0.01

        # Observation vector
        self.z = np.array([[1], [0], [0], [0]])

    def set_obj(self, quaternion):
        self.quaternion = quaternion
        self.quaternion_list = [self.X]

        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X)
        self.X_list = [euler_angles.squeeze().tolist()]

    def normalize_quaternion(self, q):
        mag = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        return q / mag

    def quaternion_product(self, a, b):
        """
        外積計算
        :param a: ndarray (4,1)
        :param b: ndarray (4,1)
        :return: ndarray (4,1)
        """

        a1 = a[0, 0]
        a2 = a[1, 0]
        a3 = a[2, 0]
        a4 = a[3, 0]

        b1 = b[0,0]
        b2 = b[1,0]
        b3 = b[2,0]
        b4 = b[3,0]

        p1 = a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4
        p2 = a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3
        p3 = a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2
        p4 = a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1

        P = np.array([[p1], [p2], [p3], [p4]])

        return P

    def gradient_descent(self, accel, mag, q, mu):

        q1 = q[0,0]
        q2 = q[1,0]
        q3 = q[2,0]
        q4 = q[3,0]

        i = 1

        while i <= 10 :
            fg1 = 2 * (q2 * q4 - q1 * q3) - accel[0,0]
            fg2 = 2 * (q1 * q2 + q3 * q4) - accel[1,0]
            fg3 = 2 * (0.5 - q2 ** 2 - q3 ** 2) - accel[2,0]
            fg = np.array([[fg1], [fg2], [fg3]])

            Jg1 = np.array([-2 * q3, 2 * q4, -2 * q1, 2 * q2])
            Jg2 = np.array([2 * q2, 2 * q1, 2 * q4, 2 * q3])
            Jg3 = np.array([0, -4 * q2, -4 * q3, 0])
            Jg = np.array([[Jg1], [Jg2], [Jg3]]).squeeze()

            mag = mag / np.linalg.norm(mag)

            q_coniug = np.array([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])     #共役複素数

            hTemp = self.quaternion_product(q, np.array([[0], [mag[0,0]], [mag[1,0]], [mag[2,0]]]))
            h = self.quaternion_product(hTemp, q_coniug)

            # b = np.array([[math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [0], [h[3,0]]])
            b = np.array([[0], [math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [h[3,0]]])
            b = b / np.linalg.norm(b)

            #b = np.array([[0], [0.653], [-0.758]])

            # fb1 = 2 * b[0,0] * (0.5 - q3 ** 2 - q4 ** 2) + 2 * b[2,0] * (q2 * q4 - q1 * q3) - mag[0,0]
            # fb2 = 2 * b[0,0] * (q2 * q3 - q1 * q4) + 2 * b[2,0] * (q1 * q2 + q3 * q4) - mag[1,0]
            # fb3 = 2 * b[0,0] * (q1 * q3 + q2 * q4) + 2 * b[2,0] * (0.5 - q2 ** 2 - q3 ** 2) - mag[2,0]
            # fb = np.array([[fb1], [fb2], [fb3]])
            #
            # Jb1 = np.array([-2 * b[2,0] * q3, 2 * b[2,0] * q4,  -4 * b[0,0] * q3 - 2 * b[2,0] * q1, -4 * b[0,0] * q4 + 2 * b[2,0] * q2])
            # Jb2 = np.array([-2 * b[0,0] * q4 + 2 * b[2,0] * q2, 2 * b[0,0] * q3 + 2 * b[2,0] * q1, 2 * b[0,0] * q2 + 2 * b[2,0] * q4, -2 * b[0,0] * q1 + 2 * b[2,0] * q3])
            # Jb3 = np.array([2 * b[0,0] * q3, 2 * b[0,0] * q4 - 4 * b[2,0] * q2, 2 * b[0,0] * q1 - 4 * b[2,0] * q3, 2 * b[0,0] * q2])
            # Jb = np.array([[Jb1], [Jb2], [Jb3]]).squeeze()

            fb1 = 2 * b[1,0] * (q1*q4 + q2*q3) + 2 * b[2,0] * (q2 * q4 - q1 * q3) - mag[0,0]
            fb2 = 2 * b[1,0] * (0.5 - q2 **2 - q4 **2 ) + 2 * b[2,0] * (q1 * q2 + q3 * q4) - mag[1,0]
            fb3 = 2 * b[1,0] * (q3 * q4 - q1 * q2) + 2 * b[2,0] * (0.5 - q2 ** 2 - q3 ** 2) - mag[2,0]
            fb = np.array([[fb1], [fb2], [fb3]])

            Jb1 = np.array([2*b[1,0]*q4 -2 * b[2,0] * q3, 2*b[1,0]*q3 + 2 * b[2,0] * q4,  2 * b[1,0] * q2 - 2 * b[2,0] * q1, 2 * b[1,0] * q1 + 2 * b[2,0] * q2])
            Jb2 = np.array([2 * b[2,0] * q2, -4 * b[1,0] * q2 + 2 * b[2,0] * q1, 2 * b[2,0] * q4, -4 * b[1,0] * q4 + 2 * b[2,0] * q3])
            Jb3 = np.array([-2 * b[1,0] * q2, -2 * b[1,0] * q1 - 4 * b[2,0] * q2, 2 * b[1,0] * q4 - 4 * b[2,0] * q3, 2 * b[1,0] * q3])
            Jb = np.array([[Jb1], [Jb2], [Jb3]]).squeeze()

            fgb = np.concatenate([fg, fb])
            Jgb = np.concatenate([Jg, Jb])

            Df = np.dot(Jgb.transpose(), fgb)

            q_Temp = q - mu * Df / np.linalg.norm(Df)

            q_result = np.array([[q_Temp[0,0]],[q_Temp[1,0]],[q_Temp[2,0]],[q_Temp[3,0]]])

            q_result = q_result / np.linalg.norm(q_result)

            q1 = q_result[0, 0]
            q2 = q_result[1, 0]
            q3 = q_result[2, 0]
            q4 = q_result[3, 0]

            q = np.array([[q1], [q2], [q3], [q4]])

            i = i + 1

        return q

    def update(self, w, a, m, dt):
        self.dt = dt
        w = np.array(w)[:, np.newaxis]
        a = np.array(a)[:, np.newaxis]
        m = np.array(m)[:, np.newaxis]

        a = a / np.linalg.norm(a)
        m = m / np.linalg.norm(m)

        # Predicted states matrix (X_p = A.X + B.U + W)
        const = self.dt / 2
        F1 = np.array([1, -const * w[0,0], -const * w[1,0], -const * w[2,0]])
        F2 = np.array([const * w[0,0], 1, const * w[2,0], -const * w[1,0]])
        F3 = np.array([const * w[1,0], -const * w[2,0], 1, const * w[0,0]])
        F4 = np.array([const * w[2,0], const * w[1,0], -const * w[0,0], 1])
        F = np.array([[F1], [F2], [F3], [F4]]).squeeze()

        self.X_p = np.dot(F, self.X)

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(F, self.P), F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        K = np.dot(np.dot(self.P_p, self.H.transpose()),
                   np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        # Current observations matrix (Y = C.Y + Z)
        # dq = 0.5 * self.quaternion_product(a=self.X, b=np.array([[0], [w[0,0]], [w[1,0]], [w[2,0]]]))
        # dqnorm = np.linalg.norm(dq)
        # mu = 1 * dqnorm * self.dt

        mu0 = 0.01
        beta = 0.1
        omega = np.linalg.norm(w)
        mu = mu0 + beta * omega * self.dt

        self.z = self.gradient_descent(a, m, self.z, mu)
        self.z = self.z / np.linalg.norm(self.z)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(K, self.z - np.dot(self.H, self.X_p))
        self.X = self.normalize_quaternion(self.X)

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P_p)

        # Information to keep
        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X)

        self.quaternion_list.append(self.X)
        self.X_list.append(euler_angles.squeeze().tolist())
class Gauss_Newton_Kalman_Filter():

    def __init__(self, sequence):
        # Initial values
        self.sequence = sequence
        self.X = np.array([[1], [0], [0], [0]])
        self.P = np.identity(4) * 2

        # the process covariance matrix
        # var = np.array([(0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2])
        # Q1 = [var[0] + var[1] + var[2], -var[0] + var[1] - var[2], -var[0] - var[1] + var[2], var[0] - var[2] - var[2]]
        # Q2 = [-var[0] + var[1] - var[2], var[0] + var[1] + var[2], var[0] - var[1] - var[2], -var[0] - var[2] + var[2]]
        # Q3 = [-var[0] - var[1] + var[2], var[0] - var[1] - var[2], var[0] + var[1] + var[2], -var[0] + var[2] - var[2]]
        # Q4 = [var[0] - var[1] - var[2], -var[0] - var[1] + var[2], -var[0] + var[1] - var[2], var[0] + var[2] + var[2]]
        # self.Q = np.array([[Q1], [Q2], [Q3], [Q4]])

        self.Q = np.identity(4) * 0.01

        # the transformation matrix (obsevation　→　state)
        self.H = np.identity(4)

        # the identity matrix
        self.I = np.identity(4)

        # the observations covariance matrix
        self.R = np.identity(4) * 0.01

        # Observation vector
        self.z = np.array([[1], [0], [0], [0]])

    def set_obj(self, quaternion):
        self.quaternion = quaternion
        self.quaternion_list = [self.X]

        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X)
        self.X_list = [euler_angles.squeeze().tolist()]


    def normalize_quaternion(self, q):
        mag = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        return q / mag

    def quaternion_product(self, a, b):
        """
        外積計算
        :param a: ndarray (4,1)
        :param b: ndarray (4,1)
        :return: ndarray (4,1)
        """

        a1 = a[0, 0]
        a2 = a[1, 0]
        a3 = a[2, 0]
        a4 = a[3, 0]

        b1 = b[0,0]
        b2 = b[1,0]
        b3 = b[2,0]
        b4 = b[3,0]

        p1 = a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4
        p2 = a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3
        p3 = a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2
        p4 = a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1

        P = np.array([[p1], [p2], [p3], [p4]])

        return P

    def compute_jacobian(self, a,b,c,d,Ax,Ay,Az,Mx,My,Mz):

        J11 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J12 = -2 * b * Ax + 2 * a * Ay + 2 * d * Az
        J13 = -2 * c * Ax - 2 * d * Ay + 2 * a * Az
        J14 = 2 * d * Ax - 2 * c * Ay + 2 * b * Az

        J21 = 2 * b * Ax - 2 * a * Ay - 2 * d * Az
        J22 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J23 = 2 * d * Ax - 2 * c * Ay + 2 * b * Az
        J24 = 2 * c * Ax + 2 * d * Ay - 2 * a * Az

        J31 = 2 * c * Ax + 2 * d * Ay - 2 * a * Az
        J32 = -2 * d * Ax + 2 * c * Ay - 2 * b * Az
        J33 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J34 = -2 * b * Ax + 2 * a * Ay + 2 * d * Az

        J41 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J42 = -2 * b * Mx + 2 * a * My + 2 * Mz * d
        J43 = -2 * c * Mx - 2 * d * My + 2 * a * Mz
        J44 = 2 * d * Mx - 2 * c * My + 2 * b * Mz

        J51 = 2 * b * Mx - 2 * a * My - 2 * d * Mz
        J52 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J53 = 2 * d * Mx - 2 * c * My + 2 * b * Mz
        J54 = 2 * c * Mx + 2 * d * My - 2 * a * Mz

        J61 = 2 * c * Mx + 2 * d * My - 2 * a * Mz
        J62 = -2 * d * Mx + 2 * c * My - 2 * b * Mz
        J63 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J64 = -2 * b * Mx + 2 * a * My + 2 * d * Mz

        J = -np.array([[J11, J12, J13, J14],[J21,J22,J23,J24],[J31,J32,J33,J34],[J41,J42,J43,J44],[J51,J52,J53,J54],[J61,J62,J63,J64]])

        return J

    def compute_matrix(self,a,b,c,d):

        R11 = d ** 2 + a ** 2 - b ** 2 - c ** 2
        R12 = 2 * (a * b - c * d)
        R13 = 2 * (a * c + b * d)
        R21 = 2 * (a * b + c * d)
        R22 = d ** 2 + b ** 2 - a ** 2 - c ** 2
        R23 = 2 * (b * c - a * d)
        R31 = 2 * (a * c - b * d)
        R32 = 2 * (b * c + a * d)
        R33 = d ** 2 + c ** 2 - b ** 2 - a ** 2

        R = np.array([[R11, R12, R13],[R21,R22,R23],[R31,R32,R33]])

        zero = np.zeros(shape=(3, 3))

        tmp1 = np.concatenate([R, zero], axis=1)
        tmp2 = np.concatenate([zero, R], axis=1)

        M = np.concatenate([tmp1, tmp2], axis=0)

        return M

    def gauss_newton(self, q, accel, mag):

        a = q[1,0]
        b = q[2,0]
        c = q[3,0]
        d = q[0,0]

        i = 1
        n_k = np.array([[a], [b], [c], [d]])

        n = np.array([[0],[0],[0],[0]])

        while i <= 3 :
            mag = mag / np.linalg.norm(mag)
            q_coniug = np.array([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])     #共役複素数
            hTemp = self.quaternion_product(q, np.array([[0], [mag[0,0]], [mag[1,0]], [mag[2,0]]]))
            h = self.quaternion_product(hTemp, q_coniug)

            #b = np.array([[math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [0], [h[3,0]]])
            bmag = np.array([[0], [math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [h[3,0]]])
            bmag = bmag / np.linalg.norm(bmag)

            j_nk = self.compute_jacobian(a, b, c, d, accel[0,0], accel[1,0], accel[2,0], mag[0,0], mag[1,0], mag[2,0])

            M = self.compute_matrix(a, b, c, d)

            y_e = np.concatenate([np.array([[0], [0], [1]]), bmag], axis=0)

            y_b = np.concatenate([accel, mag], axis=0)

            temp1 = np.linalg.inv(np.dot(j_nk.transpose(), j_nk))
            temp2 = np.dot(temp1, j_nk.transpose())
            temp3 = np.dot(M, y_b)
            n = n_k - np.dot(temp2, (y_e - temp3))
            n = n / np.linalg.norm(n)

            a = n[0, 0]
            b = n[1, 0]
            c = n[2, 0]
            d = n[3, 0]
            n_k = n
            q = np.array([[d], [a], [b], [c]])

            i = i + 1

        n = np.array([[n[3,0]], [n[0,0]], [n[1,0]], [n[2,0]]])

        return n

    def update(self, w, a, m, dt):
        self.dt = dt
        w = np.array(w)[:, np.newaxis]
        a = np.array(a)[:, np.newaxis]
        m = np.array(m)[:, np.newaxis]

        a = a / np.linalg.norm(a)
        m = m / np.linalg.norm(m)

        # Predicted states matrix (X_p = A.X + B.U + W)
        const = self.dt / 2
        F1 = np.array([1, -const * w[0,0], -const * w[1,0], -const * w[2,0]])
        F2 = np.array([const * w[0,0], 1, const * w[2,0], -const * w[1,0]])
        F3 = np.array([const * w[1,0], -const * w[2,0], 1, const * w[0,0]])
        F4 = np.array([const * w[2,0], const * w[1,0], -const * w[0,0], 1])
        F = np.array([[F1], [F2], [F3], [F4]]).squeeze()

        self.X_p = np.dot(F, self.X)

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(F, self.P), F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        K = np.dot(np.dot(self.P_p, self.H.transpose()),
                   np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        # Current observations matrix (Y = C.Y + Z)
        self.z = self.gauss_newton(self.z, a, m)
        self.z = self.z / np.linalg.norm(self.z)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(K, self.z - np.dot(self.H, self.X_p))
        self.X = self.normalize_quaternion(self.X)

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P_p)

        # Information to keep
        euler_angles = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(sequence=self.sequence, q=self.X)

        self.quaternion_list.append(self.X)
        self.X_list.append(euler_angles.squeeze().tolist())
def get_euler_angle(data):
    roll = []
    pitch = []
    yaw = []

    for i in range(len(data)):
        roll.append(data[i][0])
        pitch.append(data[i][1])
        yaw.append(data[i][2])

    roll = pd.DataFrame(roll)
    pitch = pd.DataFrame(pitch)
    yaw = pd.DataFrame(yaw)

    return roll, pitch, yaw
def get_the_estimated_euler_angle(sequence, KF, shimmer_name, a_cal, w_cal, m_cal):

    if KF == 'GNKF':
        #  --------  Class(Kalman filter) instantiation  --------
        gnkf = Gauss_Newton_Kalman_Filter(sequence)
        gnkf.set_obj(quaternions[shimmer_name])

        #  --------  Kalman filter inplimentation  --------
        n = len(a_cal)

        for i in range(1, n):
            a = a_cal.iloc[i,:]
            w = w_cal.iloc[i, :]
            m = m_cal.iloc[i,:]

            gnkf.update(w=w, a=a, m=m, dt=dt)

        #  --------  get the estimated euler angle by Kalman Filter --------
        roll, pitch, yaw = get_euler_angle(gnkf.X_list)

    elif KF == 'ACKF':
        #  --------  Class(Kalman filter) instantiation  --------
        ackf = Adaptive_Complementary_Kalman_Filter(dt, sequence)
        ackf.set_obj(quaternions[shimmer_name])

        #  --------  Kalman filter inplimentation  --------
        n = len(a_cal)

        for i in range(1, n):
            a = a_cal.iloc[i, :]
            w = w_cal.iloc[i, :]
            m = m_cal.iloc[i, :]

            ackf.main(w=w, a=a, m=m)

        #  --------  get the estimated euler angle by Kalman Filter --------
        roll, pitch, yaw = get_euler_angle(ackf.X_list)

    elif KF == 'EKF':
        #  --------  Class(Kalman filter) instantiation  --------
        ekf = Extended_kalman_filter(dt, sequence)
        ekf.set_obj(quaternions[shimmer_name])

        #  --------  Kalman filter inplimentation  --------
        n = len(a_cal)

        for i in range(1, n):
            a = a_cal.iloc[i, :]
            w = w_cal.iloc[i, :]
            m = m_cal.iloc[i, :]

            ekf.update(U=w, a=a, m=m, dt=dt)

        #  --------  get the estimated euler angle by Kalman Filter --------
        roll, pitch, yaw = get_euler_angle(ekf.X_list)

    else:
        #  --------  Class(Kalman filter) instantiation  --------
        gdkf = Gradient_Descent_Kalman_Filter(sequence)
        gdkf.set_obj(quaternions[shimmer_name])

        #  --------  Kalman filter inplimentation  --------
        n = len(a_cal)

        for i in range(1, n):
            a = a_cal.iloc[i, :]
            w = w_cal.iloc[i, :]
            m = m_cal.iloc[i, :]

            gdkf.update(w=w, a=a, m=m, dt=dt)

        #  --------  get the estimated euler angle by Kalman Filter --------
        roll, pitch, yaw = get_euler_angle(gdkf.X_list)

    # -------- Filtering of the estimated data --------
    euler = pd.concat([roll, pitch, yaw], axis=1, ignore_index=True)

    roll = euler.iloc[:,0]
    pitch = euler.iloc[:,1]
    yaw = euler.iloc[:,2]

    return roll, pitch, yaw
def get_evaluation_data(roll, pitch, yaw, roll_OMC, pitch_OMC, yaw_OMC):
    roll.columns = ['roll']
    pitch.columns = ['pitch']
    yaw.columns = ['yaw']

    evaluation_data = pd.concat([roll, roll_OMC, pitch, pitch_OMC, yaw, yaw_OMC],axis=1)
    evaluation_data.columns = ['roll', 'roll_OMC', 'pitch', 'pitch_OMC', 'yaw', 'yaw_OMC']

    return evaluation_data
def calculate_RMSE(data):
    """
    Calculate the Root Mean Square Error

    Parameters :
        data - DataFrame (dim: ,6):

    Returns:
        DataFrame (dim: 1,3):
    """
    roll_standard = np.array(data.loc[:, 'roll_OMC'])
    roll_pred = np.array(data.loc[:, 'roll'])
    pitch_standard = np.array(data.loc[:, 'pitch_OMC'])
    pitch_pred = np.array(data.loc[:, 'pitch'])
    yaw_standard = np.array(data.loc[:, 'yaw_OMC'])
    yaw_pred = np.array(data.loc[:, 'yaw'])

    rmse_roll = np.sqrt(mean_squared_error(roll_standard, roll_pred))
    rmse_pitch = np.sqrt(mean_squared_error(pitch_standard, pitch_pred))
    rmse_yaw = np.sqrt(mean_squared_error(yaw_standard, yaw_pred))

    result = pd.DataFrame([rmse_roll, rmse_pitch, rmse_yaw])

    return result
def get_omc_reference(sequence):
    if sequence == 'XZY':
        psi_ref = 88.91734
        theta_ref = 1.72953
        phi_ref = -37.02261

    elif sequence == 'XYZ':
        psi_ref = 88.36781
        theta_ref = -0.05646
        phi_ref = 0.11961

    elif sequence == 'YXZ':
        psi_ref = -90.32970
        theta_ref = 52.89254
        phi_ref = -88.27095

    elif sequence == 'YZX':
        psi_ref = -2.47100
        theta_ref = -36.98605
        phi_ref = 88.66921

    elif sequence == 'ZYX':
        psi_ref = -37.01253
        theta_ref = -1.97298
        phi_ref = 90.15686

    else:  # ZXY #測定不可
        psi_ref = -37
        theta_ref = 90
        phi_ref = 0

    R = Convertion_angles_rotation_matrix.convert_euler_angles_to_R(sequence=sequence, psi=psi_ref, theta=theta_ref,
                                                                    phi=phi_ref)

    return R
#-----------------------------------------------------------------------------------------------------------------------
#                                　　　　　　　　　　 Shimmerデータ処理
#-----------------------------------------------------------------------------------------------------------------------

# --------  Class instantiation  --------
data_info = Data_info(IMU_names=None)
IMU_names = data_info.IMU_names

#  --------  Select the sampling frequency  --------
frequency = data_info.IMU_frequencies[3] #100Hz
dt = 1 / frequency
sequence = 'XYZ'

#  Create a dictionary of quaternion object (one for each IMU)
quaternions = {}

for IMU_name in IMU_names:
    quaternion = Quaternion()
    quaternions[IMU_name] = quaternion

calibration = Calibration()
calibration.set_obj(data_info=data_info)

a_cal_U, w_cal_U, m_cal_U = get_the_calibrated_output('B532', data_info, calibration)
a_cal_F, w_cal_F, m_cal_F = get_the_calibrated_output('A833', data_info, calibration)

plt.figure(1)
plt.plot(a_cal_U, label='acc')
plt.plot(w_cal_U, label='gyr')
plt.plot(m_cal_U, label='mag')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(a_cal_F, label='acc')
plt.plot(w_cal_F, label='gyr')
plt.plot(m_cal_F, label='mag')
plt.legend()
plt.show()

roll_U, pitch_U, yaw_U = get_the_estimated_euler_angle(sequence, 'GNKF', 'B532', a_cal_U, w_cal_U, m_cal_U)
roll_F, pitch_F, yaw_F = get_the_estimated_euler_angle(sequence, 'GNKF', 'A833', a_cal_F, w_cal_F, m_cal_F)

# ---------------- Cut the extra part of the data ---------------------------
roll_U = roll_U[500:6000]
pitch_U = pitch_U[500:6000]
yaw_U = yaw_U[500:6000]

roll_F = roll_F[500:6000]
pitch_F = pitch_F[500:6000]
yaw_F = yaw_F[500:6000]

roll_U.reset_index(drop=True, inplace=True)
pitch_U.reset_index(drop=True, inplace=True)
yaw_U.reset_index(drop=True, inplace=True)

roll_F.reset_index(drop=True, inplace=True)
pitch_F.reset_index(drop=True, inplace=True)
yaw_F.reset_index(drop=True, inplace=True)

plt.figure(3)
plt.plot(roll_U, label='Roll')
plt.plot(pitch_U, label='Pitch')
plt.plot(yaw_U, label='Yaw')
plt.legend()
plt.show()

plt.figure(4)
plt.plot(roll_F, label='Roll')
plt.plot(pitch_F, label='Pitch')
plt.plot(yaw_F, label='Yaw')
plt.legend()
plt.show()

imu_ref_R_global = get_omc_reference(sequence)

#-----------------------------------------------------------------------------------------------------------------------
#                                　　　　　　　　　　 　     OMCデータ処理
#-----------------------------------------------------------------------------------------------------------------------

UA = CreateIKdata()
UA.run()
UA.make_unit_vector(vector_name='upper_arm')
UA.calculate_IK(parent_coordinate='base', child_coordinate=UA.vector[:, 3:])

FA = CreateIKdata()
FA.run()
FA.make_unit_vector(vector_name='forearm')
FA.calculate_IK(parent_coordinate='base', child_coordinate=FA.vector[:, 3:])

UA_D = CreateIKdata()
UA_D.run()
UA_D.make_unit_vector(vector_name='upper_arm_device')
UA_D.calculate_IK(parent_coordinate='base', child_coordinate=UA_D.vector[:, 3:])

FA_D = CreateIKdata()
FA_D.run()
FA_D.make_unit_vector(vector_name='forearm_device')
FA_D.calculate_IK(parent_coordinate='base', child_coordinate=FA_D.vector[:, 3:])

#キャリブレーション　：　ボディーフレームとデバイスフレーム間の回転行列を求める
global_R_device_U_cal = UA_D.rotation[100]
global_R_device_F_cal = FA_D.rotation[100]

global_R_body_U_cal = UA.rotation[100]
global_R_body_F_cal = FA.rotation[100]

body_R_device_U = np.dot(np.linalg.inv(global_R_body_U_cal), global_R_device_U_cal)
body_R_device_F = np.dot(np.linalg.inv(global_R_body_F_cal), global_R_device_F_cal)

device_imu_R_imu_U = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # Device frame ⇔ IMU sensor frame
device_imu_R_imu_F = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Device frame ⇔ IMU sensor frame


#-----------------------------------------------------------------------------------------------------------------------
#                                　　　　　　　　　　 肘関節運動推定
#-----------------------------------------------------------------------------------------------------------------------

elbow_euler_angle_OMC = np.empty((roll_U.shape[0], 3))
elbow_euler_angle_OMC_device = np.empty((roll_U.shape[0], 3))
elbow_euler_angle_IMU = np.empty((roll_U.shape[0], 3))


length_U = roll_U.shape[0]
lenght_F = roll_F.shape[0]
if length_U >= lenght_F:
    length = lenght_F
else:
    length = length_U

for i in range(length):
    # ------------------------------------------------------------------------------------------------------------------
    #                                　　　　　　　　　　 肘関節角度計算（IMU）
    # ------------------------------------------------------------------------------------------------------------------

    #Get the IMU sensor frame

    imu_ref_R_imu_U = Convertion_angles_rotation_matrix.convert_euler_angles_to_R(sequence=sequence, psi=roll_U[i], theta=pitch_U[i], phi=yaw_U[i])
    imu_ref_R_imu_F = Convertion_angles_rotation_matrix.convert_euler_angles_to_R(sequence=sequence, psi=roll_F[i], theta=pitch_F[i], phi=yaw_F[i])

    global_R_imu_U = np.dot(np.linalg.inv(imu_ref_R_global), imu_ref_R_imu_U)
    global_R_imu_F = np.dot(np.linalg.inv(imu_ref_R_global), imu_ref_R_imu_F)

    #Get the device frame_from IMU

    global_R_device_imu_U = np.dot(global_R_imu_U, np.linalg.inv(device_imu_R_imu_U))
    global_R_device_imu_F = np.dot(global_R_imu_F, np.linalg.inv(device_imu_R_imu_F))

    global_R_body_imu_U = np.dot(global_R_device_imu_U, np.linalg.inv(body_R_device_U))
    global_R_body_imu_F = np.dot(global_R_device_imu_F, np.linalg.inv(body_R_device_F))

    #IMU
    body_U_R_body_F_IMU = np.dot(np.linalg.inv(global_R_body_imu_U), global_R_body_imu_F)
    euler_anlge_IMU = Convertion_angles_rotation_matrix.convert_R_to_euler_angles(sequence=sequence, R=body_U_R_body_F_IMU)

    elbow_euler_angle_IMU[i] = euler_anlge_IMU.T

    #-------------------------------------------------------------------------------------------------------------------
    #                                　　　　　　　　　　 肘関節角度計算（OMC）
    #-------------------------------------------------------------------------------------------------------------------

    global_R_device_U = UA_D.rotation[i]
    global_R_device_F = FA_D.rotation[i]

    global_R_body_U = UA.rotation[i]
    global_R_body_F = FA.rotation[i]


    # OMC
    body_U_R_body_F_OMC = np.dot(np.linalg.inv(global_R_body_U), global_R_body_F)
    euler_anlge_OMC = Convertion_angles_rotation_matrix.convert_R_to_euler_angles(sequence=sequence, R=body_U_R_body_F_OMC)

    device_U_R_device_F_OMC = np.dot(np.linalg.inv(global_R_device_U), global_R_device_F)
    euler_anlge_OMC_device = Convertion_angles_rotation_matrix.convert_R_to_euler_angles(sequence=sequence, R=device_U_R_device_F_OMC)

    elbow_euler_angle_OMC[i] = euler_anlge_OMC.T
    elbow_euler_angle_OMC_device[i] = euler_anlge_OMC_device.T


#-----------------------------------------------------------------------------------------------------------------------
#                                　　　　　　　　　　 運動推定結果評価
#-----------------------------------------------------------------------------------------------------------------------
elbow_euler_angle_OMC = pd.DataFrame(elbow_euler_angle_OMC)
elbow_euler_angle_OMC_device = pd.DataFrame(elbow_euler_angle_OMC_device)
elbow_euler_angle_IMU = pd.DataFrame(elbow_euler_angle_IMU)

roll_OMC = elbow_euler_angle_OMC.iloc[:, 0]
pitch_OMC = elbow_euler_angle_OMC.iloc[:, 1]
yaw_OMC = elbow_euler_angle_OMC.iloc[:, 2]

roll_OMC_device = elbow_euler_angle_OMC_device.iloc[:, 0]
pitch_OMC_device = elbow_euler_angle_OMC_device.iloc[:, 1]
yaw_OMC_device = elbow_euler_angle_OMC_device.iloc[:, 2]

roll_imu = elbow_euler_angle_IMU.iloc[:,0]
pitch_imu = elbow_euler_angle_IMU.iloc[:,1]
yaw_imu = elbow_euler_angle_IMU.iloc[:,2]

fig = plt.figure(5)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)

ax1.plot(roll_OMC, label='Roll_OMC')
ax2.plot(pitch_OMC, label='Pitch_OMC')
ax3.plot(yaw_OMC, label='Yaw_OMC')
ax1.plot(roll_imu, label='Roll_IMU')
ax2.plot(pitch_imu, label='Pitch_IMU')
ax3.plot(yaw_imu, label='Yaw_IMU')

ax1.set_title('x axis rotation')
ax2.set_title('y axis rotation')
ax3.set_title('z axis rotation')

fig.legend(['OMC', 'IMU'])
fig.show()


# -------- get the evaluation data --------
evaluation_data = get_evaluation_data(roll_imu, pitch_imu, yaw_imu, roll_OMC, pitch_OMC, yaw_OMC)

# -------- Evaluate the estimation accuracy using RMSE --------
result_RMSE = calculate_RMSE(evaluation_data)