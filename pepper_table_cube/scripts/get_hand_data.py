#!/usr/bin/env python
from test.link import Link

import os
import pandas as pd
import rospy

dirname = os.path.dirname(__file__)
cwd = os.getcwd()
FILE_NAME = 'r_hand'


if __name__ == '__main__':
  r_hand = Link("pepper::r_gripper")
  rate = rospy.Rate(25)
  
  data = r_hand.get_position()
  position = data.position
  dict = {'x': position.x, 'y': position.y, 'z': position.z}
  df = pd.DataFrame(dict, index=[0, ])
  count = 0
  try:
    while not rospy.is_shutdown():
      position = data.position
      print(position)
      df.loc[count] = [position.x, position.y, position.z]         
      count += 1                
      rate.sleep()

  except rospy.ROSInterruptException:
    df.to_csv(dirname + '/test/data/{}.csv'.format(FILE_NAME), index=False)
  
