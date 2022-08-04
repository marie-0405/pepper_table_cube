import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R


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
    
  def calculate_euler_angles(self, from_rigid, to_rigid):
    from_gloabal_rotation_arrays = self.get_global_rotation_arrays(from_rigid)
    to_global_rotation_arrays = self.get_global_rotation_arrays(to_rigid)
    euler_angles = np.empty((len(from_gloabal_rotation_arrays), 3))

    for i in range(len(from_gloabal_rotation_arrays)):
      from_to_rotation_array = np.dot(np.linalg.inv(from_gloabal_rotation_arrays[i]), to_global_rotation_arrays[i])
      euler_angles[i] = R.from_matrix(from_to_rotation_array).as_euler(seq='XYZ', degrees=True)
    # print(euler_angles)
    return euler_angles

  def get_global_rotation_arrays(self, label):
    """
    回転行列を取得する
    get rotation array
    """
    labels = [label]
    for i in range(1, 3):
      labels.append("{}.{}".format(label, i))

    rigid_df = self.df[labels][3:]
    rigid_df = rigid_df.astype(float)
    rigid_euler_angles = rigid_df.values
    rigid_rotation_arrays = R.from_euler(seq='XYZ', angles=rigid_euler_angles, degrees=False).as_matrix()
    print(rigid_rotation_arrays[0])
    return rigid_rotation_arrays


if __name__ == '__main__':
  motion_capture_data_controller = MotionCaptureDataController()
  print(motion_capture_data_controller.calculate_angles("UpperArm", "ForeArm")[1000])
