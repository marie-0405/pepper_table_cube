#!/usr/bin/env python

import os
from subprocess import check_call

import actionlib_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander import MoveGroupCommander, conversions
from rospkg import RosPack
import roslib
import rospy
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from tf.transformations import quaternion_from_euler
import time

import roslib; 
roslib.load_manifest('denso_launch')
roslib.load_manifest('pr2_controllers_msgs')

import rospy, actionlib
from pr2_controllers_msgs.msg import *

rospy.init_node("action_pepper_moveit")

move_group = "right_arm"
time.sleep(5)
arm = MoveGroupCommander(move_group)
print arm.get_current_pose().pose
running_pub = rospy.Publisher("/pepper_dcm/RightArm_controller/follow_joint_trajectory", JointTrajectory)
cancel_pub = rospy.Publisher("/move_group/cancel", actionlib_msgs.msg.GoalID)


def demo() :
  for p in [[ 0.35, 0.35, 0.1],
            [ 0.3,  0.2, 0.1],]:
    print "set_pose_target(", p, ")"
    pose = PoseStamped(header = rospy.Header(stamp = rospy.Time.now(), frame_id = '/torso'),
                        pose = Pose(position = Point(*p),
                        orientation = Quaternion(*quaternion_from_euler(1.57, 0, 1.57, 'sxyz'))))
    print "pose", pose
    arm.set_pose_target(pose)
    arm.go() or arm.go() or rospy.logerr("arm.go fails")
    rospy.sleep(1)
    if rospy.is_shutdown():
        return

if __name__ == "__main__":
    while not rospy.is_shutdown():
        demo()