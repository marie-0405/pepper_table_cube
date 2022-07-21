import numpy as np
import pandas as pd
from tabulate import tabulate

# Load calibration files get from external application
class Calibration():

    def __init__(self):

        self.IMU_calibration_data = {}

    def set_obj(self, data_info):

        self.data_info = data_info

    def get(self, display=False):

        self.display = display

        for IMU_name in self.data_info.IMU_names:

            self.IMU_calibration_data[IMU_name] = {}

            offset_acc, sensitivity_acc, alignment_acc = self.get_acc_calibration_data(IMU_name=IMU_name)

            self.IMU_calibration_data[IMU_name]['ba'] = offset_acc
            self.IMU_calibration_data[IMU_name]['Ka'] = sensitivity_acc
            self.IMU_calibration_data[IMU_name]['Ra'] = alignment_acc

        for IMU_name in self.data_info.IMU_names:

            offset_gyr, sensitivity_gyr, alignment_gyr = self.get_gyr_calibration_data(IMU_name=IMU_name)

            self.IMU_calibration_data[IMU_name]['bg'] = offset_gyr
            self.IMU_calibration_data[IMU_name]['Kg'] = sensitivity_gyr
            self.IMU_calibration_data[IMU_name]['Rg'] = alignment_gyr

        for IMU_name in self.data_info.IMU_names:

            offset_mag, sensitivity_mag, alignment_mag = self.get_mag_calibration_data(IMU_name=IMU_name)

            self.IMU_calibration_data[IMU_name]['bm'] = offset_mag
            self.IMU_calibration_data[IMU_name]['Km'] = sensitivity_mag
            self.IMU_calibration_data[IMU_name]['Rm'] = alignment_mag

    def get_acc_calibration_data(self, IMU_name='B0DF'):

        # Accelerometer
        df = pd.read_csv(self.data_info.path_calibration + IMU_name + "_acc_1.5g.ini", sep=" = ", engine='python')
        data = df.values

        offset_acc = np.array([[data[0][0]],
                               [data[1][0]],
                               [data[2][0]]])
        sensitivity_acc = np.array([[data[3][0], 0, 0],
                                    [0, data[4][0], 0],
                                    [0, 0, data[5][0]]])
        alignment_acc = np.array([[data[6][0], data[7][0], data[8][0]],
                                [data[9][0], data[10][0], data[11][0]],
                                [data[12][0], data[13][0], data[14][0]]])

        if self.display:

            print(" ")
            print("-------------------------------- ")
            print(IMU_name + " - Accelerometer calibration ")
            print("-------------------------------- ")

            table = tabulate(offset_acc, tablefmt="psql")
            print("Offset (ba)")
            print(table)

            table = tabulate(sensitivity_acc, tablefmt="psql")
            print("Sensitivity (Ka)")
            print(table)

            table = tabulate(alignment_acc, tablefmt="psql")
            print("Alignment (Ra)")
            print(table)

        return offset_acc, sensitivity_acc, alignment_acc

    def get_gyr_calibration_data(self, IMU_name='B0DF'):

        # Accelerometer
        df = pd.read_csv(self.data_info.path_calibration + IMU_name + "_gyr.ini", sep=" = ", engine='python')
        data = df.values

        offset_gyr = np.array([[data[0][0]],
                               [data[1][0]],
                               [data[2][0]]])
        sensitivity_gyr = np.array([[data[3][0], 0, 0],
                                [0, data[4][0], 0],
                                [0, 0, data[5][0]]])
        alignment_gyr = np.array([[data[6][0], data[7][0], data[8][0]],
                                [data[9][0], data[10][0], data[11][0]],
                                [data[12][0], data[13][0], data[14][0]]])
        if self.display:

            print(" ")
            print("-------------------------------- ")
            print(IMU_name + " - Gyroscope calibration ")
            print("-------------------------------- ")

            table = tabulate(offset_gyr, tablefmt="psql")
            print("Offset (bg)")
            print(table)

            table = tabulate(sensitivity_gyr, tablefmt="psql")
            print("Sensitivity (Kg)")
            print(table)

            table = tabulate(alignment_gyr, tablefmt="psql")
            print("Alignment (Rg)")
            print(table)

        return offset_gyr, sensitivity_gyr, alignment_gyr

    def get_mag_calibration_data(self, IMU_name='B0DF'):

        # Accelerometer
        df = pd.read_csv(self.data_info.path_calibration + IMU_name + "_mag.ini", sep=" = ", engine='python')
        data = df.values

        offset_mag = np.array([[data[0][0]],
                               [data[1][0]],
                               [data[2][0]]])
        sensitivity_mag = np.array([[data[3][0], 0, 0],
                                [0, data[4][0], 0],
                                [0, 0, data[5][0]]])
        alignment_mag = np.array([[data[6][0], data[7][0], data[8][0]],
                                [data[9][0], data[10][0], data[11][0]],
                                [data[12][0], data[13][0], data[14][0]]])

        if self.display:

            print(" ")
            print("-------------------------------- ")
            print(IMU_name + " - Magnetometer calibration ")
            print("-------------------------------- ")

            table = tabulate(offset_mag, tablefmt="psql")
            print("Offset (bm)")
            print(table)

            table = tabulate(sensitivity_mag, tablefmt="psql")
            print("Sensitivity (Km)")
            print(table)

            table = tabulate(alignment_mag, tablefmt="psql")
            print("Alignment (Rm)")
            print(table)

        return offset_mag, sensitivity_mag, alignment_mag
