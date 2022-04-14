#!/usr/bin/env python

import os
from subprocess import check_call

import actionlib_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander import MoveGroupCommander, conversions, RobotCommander
from rospkg import RosPack
import roslib
import rospy
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler
import time

rospy.init_node("test_pepper_moveit")

move_group = "right_arm"
time.sleep(5)
arm = MoveGroupCommander(move_group)
robot = RobotCommander()

# Change the value of tolerance(required)
arm.set_goal_tolerance(0.01)
arm.set_goal_position_tolerance(0.01) 
arm.set_goal_orientation_tolerance(0.01)
arm.set_planning_time(5.0)

pose_goal = arm.get_current_pose().pose

print "CURRENT POSE\n", pose_goal
# print "CURRENT STATE\n", state

pose_goal.position.x = -0.24196194824
pose_goal.position.y = 0.779524193521
pose_goal.position.z = 0.886586306529
pose_goal.orientation.x = -0.0525654510206
pose_goal.orientation.y = -0.0780528755984
pose_goal.orientation.z = 0.02798770359
pose_goal.orientation.w = 0.99516898586

print "TARGET POSE\n", pose_goal


arm.set_pose_target(pose_goal)

is_shutdown = False
def demo() :
    plan = arm.go(wait=True)
    global is_shutdown
    if plan:
        # print "PLAN", arm.plan()
        is_shutdown = True
    # Calling `stop()` ensures that there is no residual movement
    # arm.stop()
    # arm.go() or arm.go() or rospy.logerr("arm.go fails")
    rospy.sleep(1)
    if is_shutdown:
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        arm.clear_pose_targets()
        return

if __name__ == "__main__":
    while not rospy.is_shutdown():
        demo()