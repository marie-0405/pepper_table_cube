import numpy as np
import os
import pandas as pd

# import rospkg

FILE_NAME = 'RightToLeft'
# rospack = rospkg.RosPack()


class MotionCaptureDataController():
  def __init__(self):
    # pkg_path = rospack.get_path('human_training')
    # self.file_path = pkg_path + '/data/' + FILE_NAME + '.csv'

    # For Windows
    current_directory = os.getcwd()
    self.file_path = current_directory + '\\..\\data\\' + FILE_NAME + '.csv'
    # print(self.file_path)

    self.df = pd.read_csv(self.file_path, header=2, index_col=0)

  def get_global_rotation_array(self, label):
    """
    回転行列を取得する
    get rotation array
    """
    labels = [label]
    for i in range(1, 3):
      labels.append("{}.{}".format(label, i))
    rigid_df = self.df[labels][3:].astype(float)
    rigid_rotation_array = rigid_df.values
    return rigid_rotation_array


if __name__ == '__main__':
  motion_capture_data_controller = MotionCaptureDataController()
  can_rotation_array = motion_capture_data_controller.get_global_rotation_array("Can")
  print(can_rotation_array)
