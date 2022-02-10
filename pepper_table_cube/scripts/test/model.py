#!/usr/bin/env python

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3
import rospy

class Model():
  def __init__(self, name):
    self.name = name
    rospy.init_node('model_state_sub', anonymous=True)
    self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.get_position )
    rospy.spin()
    
  def get_position(self, msg):
    index = msg.name.index(self.name)
    position = ModelStates()
    print(msg.pose[index])
    return msg.pose[index]

if __name__ == '__main__':   
  try:
    cube = Model("cube")
  except rospy.ROSInterruptException: pass
  