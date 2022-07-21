import os
import numpy as np

class Data_info():

    def __init__(self, IMU_names=None):

        if IMU_names is None:
            IMU_names = ["9397", "A833", "AA1B", "B4BF", "B15C", "B532", "BODF"]

        self.IMU_names = IMU_names
        self.IMU_display_name = self.IMU_names[0]  # IMU currently displayed in GUI and pygame

        self.cwd = os.getcwd()
        self.path_abs = os.path.abspath(os.path.join(self.cwd, os.pardir))
        self.path_python = self.cwd

        self.path_desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        self.path_calibration = self.path_python + "\\Data_calibrations\\"

        self.path_test = self.path_python + "\\Data\\"
        self.IMU_frequencies = [0, 10, 50, 100, 125, 166, 200]

    def current_IMU_display(self, IMU_name):

        self.IMU_display_name = IMU_name

