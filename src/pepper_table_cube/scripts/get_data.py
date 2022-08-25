#!/usr/bin/env python
from test.link import Link
from test.model import Model

import os
import pandas as pd
import rospy


FILE_NAME = 'data'
RATE = 25

dirname = os.path.dirname(__file__)
cwd = os.getcwd()


if __name__ == '__main__':
  rospy.init_node('get_data', anonymous=True)
  
  r_hand = Link("pepper::r_gripper")
  target = Model("target")
  cube = Model("cube")

  r_hand_pose = r_hand.get_position()
  target_pose = target.get_position()
  cube_pose = cube.get_position()

  rate = rospy.Rate(RATE)
  
  dict = {'cube_x': cube_pose.x,'cube_y': cube_pose.y,
          'hand_x': r_hand_pose.x, 'hand_y': r_hand_pose.y, 
          'target_x': target_pose.x,'target_y': target_pose.y,
          }
  df = pd.DataFrame(dict, index=[0, ])
  count = 0

  try:
    while not rospy.is_shutdown():
      count += 1 
      r_hand_pose = r_hand.get_position()
      target_pose = target.get_position()
      cube_pose = cube.get_position()
      
      df.loc[count] = [cube_pose.x, cube_pose.y, r_hand_pose.x, r_hand_pose.y, target_pose.x, target_pose.y]                       
      rate.sleep()

  except rospy.ROSInterruptException:
    df.to_csv(dirname + '/test/data/{}.csv'.format(FILE_NAME), index=False)
  
