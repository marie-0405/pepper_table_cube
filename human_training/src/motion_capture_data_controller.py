import pandas as pd

import rospkg

FILE_NAME = 'RightToLeft'
rospack = rospkg.RosPack()


class MotionCaptureDataController():
  def __init__(self):
    pkg_path = rospack.get_path('human')
    self.file_path = pkg_path + '/motion_capture_data/' + FILE_NAME + '.csv'
    self.label = [
      'MarkerSet:CAN1', 'MarkerSet:CAN1.1', 'MarkerSet:CAN1.2',
      'MarkerSet:CAN2', 'MarkerSet:CAN2.1', 'MarkerSet:CAN2.2',
      'MarkerSet:CAN3', 'MarkerSet:CAN3.1', 'MarkerSet:CAN3.2',
      'MarkerSet:CAN4', 'MarkerSet:CAN4.1', 'MarkerSet:CAN4.2',
      'MarkerSet:CV7', 'MarkerSet:CV7.1', 'MarkerSet:CV7.2',
      'MarkerSet:RCAJ', 'MarkerSet:RCAJ.1', 'MarkerSet:RCAJ.2',
      'MarkerSet:RHLE', 'MarkerSet:RHLE.1', 'MarkerSet:RHLE.2',
      'MarkerSet:RHME', 'MarkerSet:RHME.1', 'MarkerSet:RHME.2',
      'MarkerSet:RIND', 'MarkerSet:RIND.1', 'MarkerSet:RIND.2',
      'MarkerSet:RPLM', 'MarkerSet:RPLM.1', 'MarkerSet:RPLM.2',
      'MarkerSet:RPNK', 'MarkerSet:RPNK.1', 'MarkerSet:RPNK.2',
      'MarkerSet:RRSP', 'MarkerSet:RRSP.1', 'MarkerSet:RRSP.2',
      'MarkerSet:RUSP', 'MarkerSet:RUSP.1', 'MarkerSet:RUSP.2',
      'MarkerSet:SJN', 'MarkerSet:SJN.1', 'MarkerSet:SJN.2',
      'MarkerSet:SXS', 'MarkerSet:SXS.1', 'MarkerSet:SXS.2',
      'MarkerSet:TLL', 'MarkerSet:TLL.1', 'MarkerSet:TLL.2',
      'MarkerSet:TLR', 'MarkerSet:TLR.1', 'MarkerSet:TLR.2',
      'MarkerSet:TUL', 'MarkerSet:TUL.1', 'MarkerSet:TUL.2',
      'MarkerSet:TUR', 'MarkerSet:TUR.1', 'MarkerSet:TUR.2',
      'MarkerSet:TV7', 'MarkerSet:TV7.1', 'MarkerSet:TV7.2'
    ]

  def read(self):
    df = pd.read_csv(self.file_path, header=3, index_col=0)
    print(df)
    print(df.columns)
    print(df.index)
    print(df['Name'])
    # print(df.loc['0'])

if __name__ == '__main__':
  motion_capture_data_controller = MotionCaptureDataController()
  motion_capture_data_controller.read()
