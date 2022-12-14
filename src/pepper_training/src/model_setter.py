#!/usr/bin/env python

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
import rospy

class ModelSetter():
  def __init__(self, model_name):
    self.model_name = model_name
    self.service = '/gazebo/set_model_state'
    self.server = rospy.ServiceProxy(self.service, SetModelState)

  def set_position(self, x, y, z):
    msg = ModelState()
    msg.model_name = self.model_name
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z

    result = self.server(msg)
    print(result)

if __name__ == '__main__':   
  try:
    rospy.init_node('model_setter_service', anonymous=True)
    cube = ModelSetter("cube")
    cube.set_position(0.3, -0.28, 0.73)

  except rospy.ROSInterruptException: pass
  