#!/usr/bin/env python
# coding: UTF-8

'''
For confirming the joint data from human
'''
import rospy
import rospkg
import numpy as np
import time
from geometry_msgs.msg import Pose
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from body_action import BodyAction
from pepper_state_joint import PepperState

FILE_NAME = 'test2'

def main():
  rospack = rospkg.RosPack()
  pkg_path = rospack.get_path('pepper_table_cube')
  FILE_PATH = pkg_path + '/scripts/experience_data/{}_3d.csv'.format(FILE_NAME)

  joint_df = pd.read_csv(FILE_PATH)
  joint_names = ["RShoulderRoll", "RShoulderPitch", "RElbowYaw", "RElbowRoll", "RWristYaw"]

  joint_df = joint_df.loc[:, joint_names]  # Get right arm joint data
  joint_df.loc[:, 'RWristYaw'] = 0.0  # Set 0 to all of RWristYaw

  positions = joint_df.values
  positions = positions[:-1]  # Remove last values because including zero value

  joint_names = ["RShoulderRoll", "RShoulderPitch","RElbowYaw", "RElbowRoll", "RWristYaw"]
  pepper_body_action_object = BodyAction(joint_names, "/pepper_dcm/RightArm_controller/follow_joint_trajectory")
  
  for next_positions in positions:
    # Get current positions of joint
        current_positions = self.pepper_state_object.get_joint_positions(self.joint_names)
        # Then we send the command to the robot and let it go
        self.pepper_body_action_object.move_joints(current_positions, next_positions)