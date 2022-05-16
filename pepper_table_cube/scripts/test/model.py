#!/usr/bin/env python

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3
import rospy
import time

class Model():
  def __init__(self, name):
    self.name = name
    self.topic = '/gazebo/model_states'

    # rospy.Subscriber("/gazebo/model_states", ModelStates, self.models_state_callback)
    msg = rospy.wait_for_message(self.topic, ModelStates)
    self.index = msg.name.index(self.name)
    # rospy.spin()
  
  def models_state_callback(self, msg):
    self.models_state = msg
    self.index = self.models_state.name.index(self.name)

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
  