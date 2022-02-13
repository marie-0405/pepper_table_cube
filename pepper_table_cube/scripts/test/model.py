#!/usr/bin/env python

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3
import rospy
import time

class Model():
  def __init__(self, name):
    self.name = name
    self.topic = '/gazebo/model_states'

    msg = rospy.wait_for_message(self.topic, ModelStates)
    self.index = msg.name.index(self.name)
  
  def get_position(self):
    msg = rospy.wait_for_message(self.topic, ModelStates)
    return msg.pose[self.index].position
  
  def get_twist(self, msg):
    msg = rospy.wait_for_message(self.topic, ModelStates)
    return msg.twist[self.index]

if __name__ == '__main__':   
  try:
    rospy.init_node('model_state_sub', anonymous=True)
    cube = Model("cube")
    print(cube.get_position())
  except rospy.ROSInterruptException: pass
  