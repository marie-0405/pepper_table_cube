# import rapidjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt
from Classes.IMU_Attitude import Convertion_angles_rotation_matrix
np.set_printoptions(suppress=True)

class Parse:

    @staticmethod
    def intTryParse(value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    @staticmethod
    def floatTryParse(value):
        try:
            return float(value), True
        except ValueError:
            return value, False

class Data_from_IMU:

    def __init__(self):
        """"""

        self.count = 0
        self.plot_count = 0

        self.IMUs_data = {}
        self.IMUs_data_c = {}
        self.IMUs_data_q = {}

        self._observers = []

        self.IMU_bodies = ["right_hand", "left_hand"]
        self.model_index = 0
        self.index_for_unity_angle = [15, 16, 17]  # Default value for sending angle data (self.set_data_index_for_unity_angle to change it)

    def set_obj(self, data_info, positions, euler_quaternions, euler_accelerometers, euler_gyroscopes, euler_CFs, euler_LKFs, euler_EKFs, euler_ACKFs, euler_GDKFs, euler_GNKFs):

        self.data_info = data_info
        self.positions = positions

        self.euler_quaternions = euler_quaternions
        self.euler_accelerometers = euler_accelerometers
        self.euler_gyroscopes = euler_gyroscopes
        self.euler_CFs = euler_CFs
        self.euler_LKFs = euler_LKFs

        self.euler_EKFs = euler_EKFs
        self.euler_ACKFs = euler_ACKFs
        self.euler_GDKFs = euler_GDKFs
        self.euler_GNKFs = euler_GNKFs

        self.IMU_names = self.data_info.IMU_names

    def process_message(self, message_client):

        pb = False

        if message_client[0] == 'd':  # bytes contain data
            # print(message_client[1:])
            # print(message_client[1:].split("d"))
            # self.message_client = message_client[1:].split("d")[0]
            self.message_client = message_client[1:]
        else:
            print("Problem in data - d")
            pb = True

        try:
            json_dict = rapidjson.loads(self.message_client)
        except:
            # print("Problem with data received: " + str(self.message_client))
            pb = True

        if pb:
            return

        if all(elem in json_dict.keys() for elem in ["time", "frequency_IMU", "frequency_client", "model_index"]):

            if not len(json_dict["time"]) == 13:  # Check time
                print("Problem in data - time")
                pb = True
            else:
                self.time = json_dict["time"]

            value, test = Parse.intTryParse(json_dict["frequency_IMU"])  # Check if data can be integer or only string
            if test:
                self.frequency_IMU = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["frequency_client"])  # Check if data can be integer or only string
            if test:
                self.frequency_client = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["model_index"])  # Check if data can be integer or only string
            if test:
                if (value != self.model_index):
                    self.set_data_index_for_unity_angle(index=value)
                    self.model_index = value
            else:
                print("Problem in data - String/Int")
                pb = True

            if not len(json_dict.keys()) > 3:  # Check the number of channel in the data
                print("Problem in data - channel")
                pb = True
        else:
            pb = True

        if pb:
            return

        for k in json_dict.keys():
            if k in self.IMU_bodies:
                channel_data = json_dict[k].split(",")
                if (len(channel_data) == 1):
                    pb = True
                else:
                    try:
                        values = list(map(float, channel_data))
                        if not k in self.IMUs_data:
                            self.IMUs_data[k] = [values]
                        else:
                            self.IMUs_data[k].append(values)
                    except ValueError:
                        self.IMUs_data[k].append([0])

        if pb:
            return

        self.calibrate_data(data=self.IMUs_data)
        self.count += 1

    def calibrate_data(self, data):

        for IMU_name in list(data.keys()):

            if IMU_name not in self.IMUs_data_c:
                self.IMUs_data_c[IMU_name] = [np.array([0 for i in range(31)])]

            data_unc = self.IMUs_data[IMU_name][-1]

            if len(data_unc) == 9:

                # Accelerometer
                a_unc = np.array(data_unc[0:3])[:, np.newaxis]
                a_c = a_unc

                # ba = self.calibration.IMU_calibration_data[IMU_name]['ba']
                # Ka = self.calibration.IMU_calibration_data[IMU_name]['Ka']
                # Ra = self.calibration.IMU_calibration_data[IMU_name]['Ra']
                #
                # temp = np.dot(np.linalg.inv(Ra), np.linalg.inv(Ka))
                # a_c = np.dot(temp, a_unc - ba)

                # Gyroscope
                w_unc = np.array(data_unc[3:6])[:, np.newaxis]
                w_c = w_unc

                # bg = self.calibration.IMU_calibration_data[IMU_name]['bg']
                # Kg = self.calibration.IMU_calibration_data[IMU_name]['Kg']
                # Rg = self.calibration.IMU_calibration_data[IMU_name]['Rg']
                #
                # temp = np.dot(np.linalg.inv(Rg), np.linalg.inv(Kg))
                # w_c = np.dot(temp, w_unc - bg)

                # Magnetometer
                m_unc = np.array(data_unc[6:9])[:, np.newaxis]
                m_c = m_unc

                # bm = self.calibration.IMU_calibration_data[IMU_name]['bm']
                # Km = self.calibration.IMU_calibration_data[IMU_name]['Km']
                # Rm = self.calibration.IMU_calibration_data[IMU_name]['Rm']
                #
                # temp = np.dot(np.linalg.inv(Rm), np.linalg.inv(Km))
                # m_c = np.dot(temp, m_unc - bm)

                # Euler quaternion
                self.euler_quaternions[IMU_name].update(w=w_c, dt=1/self.frequency_client)
                R = self.euler_quaternions[IMU_name].R
                euler_q = self.euler_quaternions[IMU_name].eulers_deg

                # Euler accelerometer
                self.euler_accelerometers[IMU_name].update(a=a_c)
                eulers_acc = self.euler_accelerometers[IMU_name].eulers_deg

                # Euler gyroscope
                self.euler_gyroscopes[IMU_name].update(w=w_c, dt=1/self.frequency_client)
                eulers_gyr = self.euler_gyroscopes[IMU_name].eulers_deg

                # Euler complementary filter
                self.euler_CFs[IMU_name].update(a=a_c, w=w_c, dt=1/self.frequency_client)
                eulers_CF = self.euler_CFs[IMU_name].eulers_deg

                # Euler LKF
                self.euler_LKFs[IMU_name].update(a=a_c, w=w_c)
                eulers_LKF = self.euler_LKFs[IMU_name].eulers_deg

                # Euler EKF
                self.euler_EKFs[IMU_name].update(a=a_c, w=w_c, m=m_c, dt=1/self.frequency_client)
                eulers_EKF = self.euler_EKFs[IMU_name].eulers_deg

                # Euler ACKF
                self.euler_ACKFs[IMU_name].update(a=a_c, w=w_c, m=m_c, dt=1/self.frequency_client)
                eulers_ACKF = self.euler_ACKFs[IMU_name].eulers_deg

                # Euler GDKF
                self.euler_GDKFs[IMU_name].update(a=a_c, w=w_c, m=m_c, dt=1/self.frequency_client)
                eulers_GDKF = self.euler_GDKFs[IMU_name].eulers_deg

                # Euler GNKF
                self.euler_GNKFs[IMU_name].update(a=a_c, w=w_c, m=m_c, dt=1/self.frequency_client)
                eulers_GNKF = self.euler_GNKFs[IMU_name].eulers_deg

                # Position
                self.positions[IMU_name].move(R, a=a_c, dt=1/self.frequency_client)

                p = self.positions[IMU_name].get_current_position()
                v = self.positions[IMU_name].get_current_velocity()
                a_earth = self.positions[IMU_name].get_current_acceleration_earth()

                # if self.count % 10 == 0:
                #
                #     print(IMU_name, " - q: " + np.array2string(data_c[15:18], precision=1, separator=","),
                #           " - acc: " + np.array2string(data_c[18:21], precision=1, separator=","),
                #           " - gyr: " + np.array2string(data_c[21:24], precision=1, separator=","),
                #           " - CF: " + np.array2string(data_c[24:27], precision=1, separator=","),
                #           " - LKF: " + np.array2string(data_c[27:31], precision=1, separator=","))

            else:

                a_c = np.array([[0], [0], [0]])
                w_c = np.array([[0], [0], [0]])
                m_c = np.array([[0], [0], [0]])

                a_earth = np.array([[0], [0], [0]])
                p = np.array([[0], [0], [0]])

                euler_q = np.array([[0], [0], [0]])
                eulers_acc = np.array([[0], [0], [0]])
                eulers_gyr = np.array([[0], [0], [0]])
                eulers_CF = np.array([[0], [0], [0]])
                eulers_LKF = np.array([[0], [0], [0]])
                eulers_EKF = np.array([[0], [0], [0]])
                eulers_ACKF = np.array([[0], [0], [0]])
                eulers_GDKF = np.array([[0], [0], [0]])
                eulers_GNKF = np.array([[0], [0], [0]])

                # if self.count % 10 == 0:
                #
                #     print(IMU_name, " - q: " + np.array2string(data_c[15:18], precision=1, separator=","),
                #           " - acc: " + np.array2string(data_c[18:21], precision=1, separator=","),
                #           " - gyr: " + np.array2string(data_c[21:24], precision=1, separator=","),
                #           " - CF: " + np.array2string(data_c[24:27], precision=1, separator=","),
                #           " - LKF: " + np.array2string(data_c[27:30], precision=1, separator=","))

            data_c = np.concatenate([a_c, w_c, m_c, a_earth, p, euler_q, eulers_acc, eulers_gyr, eulers_CF, eulers_LKF, eulers_EKF, eulers_ACKF, eulers_GDKF, eulers_GNKF]).squeeze()
            self.IMUs_data_c[IMU_name].append(data_c)

    def set_data_index_for_unity_angle(self, index=0):

        """
        Parameters:
            index - int :   0: euler_q - Quaternion
                            1: eulers_acc - Acceleromter
                            2: eulers_gyr - Gyroscope
                            3: eulers_CF - Complementary Filter
                            4: eulers_LKF - Linear Kalman Filter
                            5: eulers_EKF - Extended Kalman Filter
                            6: eulers_ACKF - Adaptive Complementary Kalman Filter
                            7: eulers_GDKF - Gradient Descent Kalman Filter
                            8: eulers_GNKF - Gauss Newton Kalman Filter
                            9: All

        """

        if index == 0:
            self.index_for_unity_angle = [15, 16, 17]
        elif index == 1:
            self.index_for_unity_angle = [18, 19, 20]
        elif index == 2:
            self.index_for_unity_angle = [21, 22, 23]
        elif index == 3:
            self.index_for_unity_angle = [24, 25, 26]
        elif index == 4:
            self.index_for_unity_angle = [27, 28, 29]
        elif index == 5:
            self.index_for_unity_angle = [30, 31, 32]
        elif index == 6:
            self.index_for_unity_angle = [33, 34, 35]
        elif index == 7:
            self.index_for_unity_angle = [36, 37, 38]
        elif index == 8:
            self.index_for_unity_angle = [39, 40, 41]
        elif index == 9:
            self.index_for_unity_angle = [i for i in range(18, 42)]
        else:
            self.index_for_unity_angle = [15, 16, 17]

        print(self.index_for_unity_angle)

    def get_data_for_unity_angle(self):

        s = "d"

        index = [9, 10, 11] # Position
        index.extend(self.index_for_unity_angle)

        if (len(self.IMUs_data_c.keys()) == 1):
            for IMU_name in list(self.IMUs_data_c.keys()):

                s += IMU_name

                for val in self.IMUs_data_c[IMU_name][-1][index].tolist():
                    if math.isnan(val):
                        s += ","
                        s += "{:.1f}".format(0)
                    else:
                        s += ","
                        s += "{:.1f}".format(val)
                s += ";"

                # Temporary simulate BODF with the same data as 9397
                for temp in ["9397"]:
                    s += temp
                    for val in self.IMUs_data_c[IMU_name][-1][index].tolist():
                        if math.isnan(val):
                            s += ","
                            s += "{:.1f}".format(0)
                        else:
                            s += ","
                            s += "{:.1f}".format(val)
                    s += ";"

        elif (len(self.IMUs_data_c.keys()) == 2):

            for IMU_name in list(self.IMUs_data_c.keys()):
                s += IMU_name
                for val in self.IMUs_data_c[IMU_name][-1][index].tolist():
                    if math.isnan(val):
                        s += ","
                        s += "{:.1f}".format(0)
                    else:
                        s += ","
                        s += "{:.1f}".format(val)
                s += ";"

        elif (len(self.IMUs_data_c.keys()) == 0):

            s = "dright_hand,0.0,0.0,0.0,1.0,0.00,0.0;left_hand,0.0,90.0,0.0,2.0,0.0,0.0;"

        return s

class Data_IMU_offline():

    def __init__(self):

        super().__init__()

        self.IMUs_data = {}
        self.IMUs_data_c = {}

        self.count = 0

    def set_obj(self, data_info, calibration):

        self.data_info = data_info
        self.calibration = calibration

    def set_IMU_name(self, IMU_name):

        self.IMU_name = IMU_name
        self.IMU_index = self.data_info.IMU_names.index(self.IMU_name)

    def set_path_file(self, path_file):

        self.path_file = path_file

    def get_data(self):

        self.df_head = pd.read_csv(self.path_file, sep="\t", nrows=8, header=None)
        self.df_IMU_info = pd.read_csv(self.path_file, sep="\t", skiprows=[i for i in range(0, 8)], nrows=1, header=None)
        self.samplingRatesIndex = self.df_IMU_info[1].values[0]

        self.df = pd.read_csv(self.path_file, skiprows=[i for i in range(0, 15)])

        self.times = self.df['TimeStamp'].values
        self.counts = len(self.times)
        self.data_raw = self.df.values[:, 1:].tolist()

        self.data_raw = np.array(self.data_raw)

    def calibrate_data(self, fs, filtering="moving"):

        a_uncal = self.df.loc[:, ['XAccel', 'YAccel', 'ZAccel']]
        w_uncal = self.df.loc[:, ['XGyro', 'YGyro', 'ZGyro']]
        m_uncal = self.df.loc[:, ['XMag', 'YMag', 'ZMag']]

        ba = self.calibration.IMU_calibration_data[self.IMU_name]['ba']
        Ka = self.calibration.IMU_calibration_data[self.IMU_name]['Ka']
        Ra = self.calibration.IMU_calibration_data[self.IMU_name]['Ra']

        temp = np.dot(np.linalg.inv(Ra), np.linalg.inv(Ka))
        a_cal = np.dot(temp, a_uncal.T - ba)

        bg = self.calibration.IMU_calibration_data[self.IMU_name]['bg']
        Kg = self.calibration.IMU_calibration_data[self.IMU_name]['Kg']
        Rg = self.calibration.IMU_calibration_data[self.IMU_name]['Rg']

        temp = np.dot(np.linalg.inv(Rg), np.linalg.inv(Kg))
        w_cal = np.dot(temp, w_uncal.T - bg)

        w_cal = Convertion_angles_rotation_matrix.deg2rad(w_cal)

        bm = self.calibration.IMU_calibration_data[self.IMU_name]['bm']
        Km = self.calibration.IMU_calibration_data[self.IMU_name]['Km']
        Rm = -self.calibration.IMU_calibration_data[self.IMU_name]['Rm']

        temp = np.dot(np.linalg.inv(Rm), np.linalg.inv(Km))
        m_cal = np.dot(temp, m_uncal.T - bm)

        self.a_cal = a_cal.transpose()
        self.w_cal = w_cal.transpose()
        self.m_cal = m_cal.transpose()

        if filtering == "moving":
            size = 10
            self.a_cal = Filtering.moving_average_filter(signal_in=self.a_cal, size=size)
            self.w_cal = Filtering.moving_average_filter(signal_in=self.w_cal, size=size)
            self.m_cal = Filtering.moving_average_filter(signal_in=self.m_cal, size=size)
        elif filtering == "low":
            self.a_cal = Filtering.butterworth_filter(signal_in=self.a_cal, cutoff=10, fs=fs, order=4)
            self.w_cal = Filtering.butterworth_filter(signal_in=self.w_cal, cutoff=10, fs=fs, order=4)
            self.m_cal = Filtering.butterworth_filter(signal_in=self.m_cal, cutoff=10, fs=fs, order=4)
        elif filtering == "exp":
            a = 0.2
            self.a_cal = Filtering.exponential_moving_average_filter(signal_in=self.a_cal, a=a)
            self.w_cal = Filtering.exponential_moving_average_filter(signal_in=self.w_cal, a=a)
            self.m_cal = Filtering.exponential_moving_average_filter(signal_in=self.m_cal, a=a)
        else:
            self.a_cal = self.a_cal
            self.w_cal = self.w_cal
            self.m_cal = self.m_cal

        self.data = np.concatenate([self.a_cal, self.w_cal, self.m_cal], axis=1)
        self.n_frame = self.data.shape[0]

    def create_message(self, count):

        if count < self.data_raw.shape[0]:

            temp = [str(v) for v in self.data_raw[count]]
            message_client = "d{\"Time\":\"" + str(self.times[count])[0:13] + "\",\"Frequency_IMU_index\":\"6\"," + "\"Frequency_client\":\"100\",\"" + str(self.IMU_name) + "\":\"" + ",".join(temp) + "\"}"

            self.process_message(message_client=message_client)

        else:
            print("Problem with count value in create_message()")

    def process_message(self, message_client):

        if message_client[0] == 'd':  # bytes contain data
            # print(message_client[1:])
            # print(message_client[1:].split("d"))
            # self.message_client = message_client[1:].split("d")[0]
            self.message_client = message_client[1:]
        else:
            print("Problem in data - d")
            return

        try:
            json_dict = rapidjson.loads(self.message_client)
        except:
            # print("Problem with data received: " + str(self.message_client))
            return

        pb = False

        if all(elem in json_dict.keys() for elem in ["time", "frequency_IMU", "frequency_client", "model_index"]):

            if not len(json_dict["time"]) == 13:  # Check time
                print("Problem in data - time")
                pb = True
            else:
                self.time = json_dict["time"]

            value, test = Parse.intTryParse(json_dict["frequency_IMU"])  # Check if data can be integer or only string
            if test:
                self.frequency_IMU = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["frequency_client"])  # Check if data can be integer or only string
            if test:
                self.frequency_client = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["model_index"])  # Check if data can be integer or only string
            if test:
                if (value != self.model_index):
                    self.set_data_index_for_unity_angle(index=value)
                    self.model_index = value
            else:
                print("Problem in data - String/Int")
                pb = True

            if not len(json_dict.keys()) > 3:  # Check the number of channel in the data
                print("Problem in data - channel")
                pb = True
        else:
            pb = True

        if pb:
            return

        for k in json_dict.keys():
            if k in self.data_info.IMU_names:
                channel_data = json_dict[k].split(",")
                if (len(channel_data) == 1):
                    return
                else:
                    try:
                        values = list(map(float, channel_data))
                        if not k in self.IMUs_data:
                            self.IMUs_data[k] = [values]
                        else:
                            self.IMUs_data[k].append(values)
                    except ValueError:
                        self.IMUs_data[k].append([0])

        self.calibrate_data(data=self.IMUs_data)
        self.count += 1

class Data_Android_offline():

    def __init__(self):

        super().__init__()

        self.IMUs_data = {}
        self.IMUs_data_c = {}

        self.count = 0

    def set_obj(self, data_info):

        self.data_info = data_info

    def set_path_file(self, path_file):

        self.path_file = path_file

    def get_data(self):

        self.df_head = pd.read_csv(self.path_file, sep="\t", nrows=0, header=None)
        self.df = pd.read_csv(self.path_file)

        self.data_raw = self.df.values.tolist()
        self.data_raw = np.array(self.data_raw)

    def calibrate_data(self, fs, filtering="moving"):

        #  --------  get the acceleration data [m/sec^2　→　m/sec^2]  --------
        a_uncal = self.df.loc[:, ['XAccel', 'YAccel', 'ZAccel']].values
        w_uncal = self.df.loc[:, ['XGyro', 'YGyro', 'ZGyro']].values
        m_uncal = self.df.loc[:, ['XMag', 'YMag', 'ZMag']].values

        self.a_cal = a_uncal.copy()
        #  --------  get the angular velocity data [deg/sec　→　rad/sec]  --------
        self.w_cal = w_uncal.copy()
        for i in range(self.w_cal.shape[0]):
            self.w_cal[i, :] = Convertion_angles_rotation_matrix.deg2rad(w_uncal[i, :])

        #  --------  get the magnetic field data [μT　→　-]  --------
        self.m_cal = m_uncal.copy()
        m_uncal_norm = np.linalg.norm(m_uncal, axis=1)

        for i in range(self.m_cal.shape[1]):
            self.m_cal[:, i] = m_uncal[:, i] / m_uncal_norm

        if filtering == "moving":
            size = 10
            self.a_cal = Filtering.moving_average_filter(signal_in=self.a_cal, size=size)
            self.w_cal = Filtering.moving_average_filter(signal_in=self.w_cal, size=size)
            self.m_cal = Filtering.moving_average_filter(signal_in=self.m_cal, size=size)
        elif filtering == "low":
            self.a_cal = Filtering.butterworth_filter(signal_in=self.a_cal, cutoff=10, fs=100, order=4)
            self.w_cal = Filtering.butterworth_filter(signal_in=self.w_cal, cutoff=10, fs=100, order=4)
            self.m_cal = Filtering.butterworth_filter(signal_in=self.m_cal, cutoff=10, fs=100, order=4)
        elif filtering == "exp":
            a = 0.2
            self.a_cal = Filtering.exponential_moving_average_filter(signal_in=self.a_cal, a=a)
            self.w_cal = Filtering.exponential_moving_average_filter(signal_in=self.w_cal, a=a)
            self.m_cal = Filtering.exponential_moving_average_filter(signal_in=self.m_cal, a=a)
        else:
            self.a_cal = self.a_cal
            self.w_cal = self.w_cal
            self.m_cal = self.m_cal

        self.data = np.concatenate([self.a_cal, self.w_cal, self.m_cal], axis=1)
        self.n_frame = self.data.shape[0]

        self.roll_smartphone = self.df.loc[:, 'roll'].values
        self.pitch_smartphone = self.df.loc[:, 'pitch'].values
        self.yaw_smartphone = self.df.loc[:, 'yaw'].values

        self.roll_smartphone = -1 * self.roll_smartphone
        self.pitch_smartphone = Unify.unify_180_to_90(self.pitch_smartphone)
        self.yaw_smartphone = self.yaw_smartphone - 180
        self.yaw_smartphone = Unify.unify_360_to_180(self.yaw_smartphone)

    def create_message(self, count):

        if count < self.data_raw.shape[0]:

            temp = [str(v) for v in self.data_raw[count]]
            message_client = "d{\"Time\":\"" + str(self.times[count])[0:13] + "\",\"Frequency_IMU_index\":\"6\"," + "\"Frequency_client\":\"100\",\"" + str(self.IMU_name) + "\":\"" + ",".join(temp) + "\"}"

            self.process_message(message_client=message_client)

        else:
            print("Problem with count value in create_message()")

    def process_message(self, message_client):

        if message_client[0] == 'd':  # bytes contain data
            # print(message_client[1:])
            # print(message_client[1:].split("d"))
            # self.message_client = message_client[1:].split("d")[0]
            self.message_client = message_client[1:]
        else:
            print("Problem in data - d")
            return

        try:
            json_dict = rapidjson.loads(self.message_client)
        except:
            # print("Problem with data received: " + str(self.message_client))
            return

        pb = False

        if all(elem in json_dict.keys() for elem in ["time", "frequency_IMU", "frequency_client", "model_index"]):

            if not len(json_dict["time"]) == 13:  # Check time
                print("Problem in data - time")
                pb = True
            else:
                self.time = json_dict["time"]

            value, test = Parse.intTryParse(json_dict["frequency_IMU"])  # Check if data can be integer or only string
            if test:
                self.frequency_IMU = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["frequency_client"])  # Check if data can be integer or only string
            if test:
                self.frequency_client = value
            else:
                print("Problem in data - String/Int")
                pb = True

            value, test = Parse.intTryParse(json_dict["model_index"])  # Check if data can be integer or only string
            if test:
                if (value != self.model_index):
                    self.set_data_index_for_unity_angle(index=value)
                    self.model_index = value
            else:
                print("Problem in data - String/Int")
                pb = True

            if not len(json_dict.keys()) > 3:  # Check the number of channel in the data
                print("Problem in data - channel")
                pb = True
        else:
            pb = True

        if pb:
            return

        for k in json_dict.keys():
            if k in self.data_info.IMU_names:
                channel_data = json_dict[k].split(",")
                if (len(channel_data) == 1):
                    return
                else:
                    try:
                        values = list(map(float, channel_data))
                        if not k in self.IMUs_data:
                            self.IMUs_data[k] = [values]
                        else:
                            self.IMUs_data[k].append(values)
                    except ValueError:
                        self.IMUs_data[k].append([0])

        self.calibrate_data(data=self.IMUs_data)
        self.count += 1

class Filtering():

    @staticmethod
    def butter_function(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff)
        return b, a

    @staticmethod
    def butter_lowpass_function(data, cutoff, fs, order=5):

        b, a = Filtering.butter_function(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butterworth_filter(signal_in, cutoff, fs, order):
        """
        Low pass filter

        Parameters :
            signal_in - np.array (dim: :,3):
            size - int:

        Returns:
            np.array (dim: :,3):
        """
        signal_out = []

        for i in range(signal_in.shape[1]):
            temp = Filtering.butter_lowpass_function(signal_in[:, i], cutoff, fs, order)[:, np.newaxis]
            signal_out.append(temp)

        signal_out = np.concatenate(signal_out, axis=1)

        return signal_out

    @staticmethod
    def moving_average_filter(signal_in, size):
        """
        Simple moving average filter

        Parameters :
            signal_in - np.array (dim: :,3):
            size - int:

        Returns:
            np.array (dim: :,3):
        """
        b = np.ones(size) / size
        signal_out = []

        for i in range(signal_in.shape[1]):
            temp = np.convolve(signal_in[:, i], b, mode='same')[:, np.newaxis]
            signal_out.append(temp)

        signal_out = np.concatenate(signal_out, axis=1)

        return signal_out

    @staticmethod
    def exponential_moving_average_filter(signal_in, a):
        """
        Exponential moving average filter (Low-pass filter)
         - extract the gravitational acceleration from the output of Accelerometer

        Parameters :
            signal_in - np.array (dim: :,3):
            a - float: coefficient

        Returns:
            np.array (dim: :,3):
        """

        signal_out = []

        for j in range(0, signal_in.shape[1]):
            temp = signal_in[:, j]
            for i in range(1, signal_in.shape[0]):
                temp[i] = a * signal_in[i, j] + (1 - a) * temp[i - 1]
            signal_out.append(temp[:, np.newaxis])

        signal_out = np.concatenate(signal_out, axis=1)

        return signal_out

class Plot:

    @staticmethod
    def plot_sensors_data(data_acc, data_gyr, data_mag, suptitle="", fig=None, ax1=None, ax2=None, ax3=None):

        if not fig and not ax1 and not ax2 and not ax3:
            fig = plt.figure(figsize=(18, 10))
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)

        ax1.set_title("Accelerometer\n[m/sec^2]")
        ax2.set_title("Gyroscope\n[rad/sec]")
        ax3.set_title("Magnetometer\n[-]")

        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        fig.suptitle(suptitle)

        for i in range(data_acc.shape[1]):
            ax1.plot(data_acc[:, i], label=labels[i], color=colors[i], linewidth=2)
            ax2.plot(data_gyr[:, i], label=labels[i], color=colors[i], linewidth=2)
            ax3.plot(data_mag[:, i], label=labels[i], color=colors[i], linewidth=2)

        ax1.grid(True)
        ax1.legend()
        ax2.grid(True)
        ax2.legend()
        ax3.grid(True)
        ax3.legend()

        fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.92, wspace=0.1, hspace=0.18)

        return fig, ax1, ax2, ax3

    @staticmethod
    def plot_eulers(data_roll, data_pitch, data_yaw, suptitle="", fig=None, ax1=None, ax2=None, ax3=None, color="r", label=""):

        if not fig and not ax1 and not ax2 and not ax3:
            fig = plt.figure(figsize=(18, 10))
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)

        ax1.set_title('(a) Roll angle (x-axis rotation)', y=-0.7, fontsize=30)
        ax2.set_title('(b) Pitch angle (y-axis rotation)', y=-0.7, fontsize=30)
        ax3.set_title('(c) Yaw angle (z-axis rotation)', y=-0.7, fontsize=30)

        ax1.set_ylim([-200, 200])
        ax2.set_ylim([-200, 200])
        ax3.set_ylim([-200, 200])

        ax1.set_ylabel('Angle[deg]', fontsize=30)
        ax2.set_ylabel('Angle[deg]', fontsize=30)
        ax3.set_ylabel('Angle[deg]', fontsize=30)

        fig.suptitle(suptitle)

        ax1.plot(data_roll, label=label, color=color, linewidth=2)
        ax2.plot(data_pitch, label=label, color=color, linewidth=2)
        ax3.plot(data_yaw, label=label, color=color, linewidth=2)

        ax1.grid(True)
        ax1.legend()
        ax2.grid(True)
        ax2.legend()
        ax3.grid(True)
        ax3.legend()

        fig.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.92, wspace=0.1, hspace=0.18)

        return fig, ax1, ax2, ax3

class Unify:

    @staticmethod
    def unify_180_to_90(data):

        if isinstance(data, list):
            data = np.array(data)
        else:
            data = data

        for i in range(len(data)):
            if data[i] >= 90:
                data[i] = 180 - data[i]
            elif data[i] < -90:
                data[i] = -180 - data[i]

        return data

    @staticmethod
    def unify_360_to_180(data):

        if isinstance(data, list):
            data = np.array(data)
        else:
            data = data

        for i in range(len(data)):
            if data[i] >= 0:
                data[i] = 180 - data[i]
            else:
                data[i] = -180 - data[i]

        return data
