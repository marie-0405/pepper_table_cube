#!/usr/bin/env python

import os
from subprocess import check_call

import actionlib_msgs.msg
import actionlib
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander import MoveGroupCommander, conversions
from rospkg import RosPack
import roslib
import rospy
import std_msgs.msg
import trajectory_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from tf.transformations import quaternion_from_euler
import os
import pandas as pd
import numpy as np
import time


import roslib 
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

FILE_NAME = 'pushing_task6'
DURATION = 0.25


class Pepper:

  def __init__(self, joint_names, client): 
    self.client = actionlib.SimpleActionClient(client, FollowJointTrajectoryAction)
    self.client.wait_for_server()
    self.goal = FollowJointTrajectoryGoal()
    self.goal.trajectory.joint_names = joint_names
    print(self.goal.trajectory.joint_names)
    
  def move(self, positions):
    """
      param: positions [[]]
    """
    points = []

    for i, position in enumerate(positions):
      # Set point
      point = trajectory_msgs.msg.JointTrajectoryPoint()
      point.positions = position
      point.time_from_start = rospy.Duration(DURATION * i)
      points.append(point)

    self.goal.trajectory.points = points
    self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)

    self.client.send_goal(self.goal)
    print ("Success" if self.client.wait_for_result() else "Failed")
  

if __name__ == '__main__':
  dirname = os.path.dirname(__file__)
  FILE_PATH = dirname + '/../../human/joint_data/{}_3d.csv'.format(FILE_NAME)

  joint_df = pd.read_csv(FILE_PATH)
  joint_names = ["RShoulderRoll", "RShoulderPitch", "RElbowYaw", "RElbowRoll", "RWristYaw"]

  joint_df = joint_df.loc[:, joint_names]  # Get right arm joint data
  joint_df.loc[:, 'RWristYaw'] = 0.0  # Set 0 to all of RWristYaw

  positions = joint_df.values
  positions = positions[:-1]  # Remove last values because including zero value

  try:
    rospy.init_node('action_pepper', anonymous=True)
    right_arm = Pepper(joint_names, "/pepper_dcm/RightArm_controller/follow_joint_trajectory")
    right_arm.move(positions)
  except rospy.ROSInterruptException: pass
  