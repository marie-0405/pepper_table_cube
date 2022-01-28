#!/usr/bin/env python

import os
from subprocess import check_call

import actionlib_msgs.msg
import actionlib
import trajectory_msgs
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander import MoveGroupCommander, conversions
from rospkg import RosPack
import roslib
import rospy
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from tf.transformations import quaternion_from_euler
import time
import numpy as np

import roslib 
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


if __name__ == '__main__':
    try:  
        rospy.init_node('action_pepper_moveit', anonymous=True)
        client = actionlib.SimpleActionClient('/pepper_dcm/RightArm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        client.wait_for_server()

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.append("RShoulderPitch")
        goal.trajectory.joint_names.append("RShoulderRoll")
        goal.trajectory.joint_names.append("RElbowYaw")
        goal.trajectory.joint_names.append("RElbowRoll")
        goal.trajectory.joint_names.append("RWristYaw")
        print goal.trajectory.joint_names

        point1 = trajectory_msgs.msg.JointTrajectoryPoint()
        point2 = trajectory_msgs.msg.JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
        point2.positions = [np.pi/6, -np.pi/8, 0.0, 0.0, 0.0]

        goal.trajectory.points = [point1, point2]

        goal.trajectory.points[0].time_from_start = rospy.Duration(2.0)
        goal.trajectory.points[1].time_from_start = rospy.Duration(4.0)

        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)

        client.send_goal(goal)
        print client.wait_for_result()
    except rospy.ROSInterruptException: pass