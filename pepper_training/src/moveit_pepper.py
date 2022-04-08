#!/usr/bin/env python
# coding: UTF-8

import actionlib
import copy
from math import pi
import rospy
import sys
import time
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import moveit_commander

DURATION = 0.25

## TODO　MoveitPepperの作成
class MoveitPepper(object):
    def __init__(self, group_name):
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this project, the group is the primary
        ## arm joints in the Pepper robot, so we set the group's name to "right_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        move_group = moveit_commander.MoveGroupCommander(group_name)  # TODO ここで処理がとまっている…？

        planning_frame = move_group.get_planning_frame()
        rospy.loginfo("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        rospy.loginfo("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        rospy.loginfo("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        rospy.loginfo("============ Printing robot state")
        rospy.loginfo(robot.get_current_state())
        rospy.loginfo("")

    def set_init_pose(self, init_pose):
        """
        Sets joints to initial position
        :return: The init Pose
        """
        # self.check_publishers_connection()
        self.move_hand(init_pose)

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._haa_joint_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _haa_joint_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_haa_joint_pub Publisher Connected")

        while (self._hfe_joint_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _hfe_joint_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_hfe_joint_pub Publisher Connected")

        while (self._kfe_joint_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _kfe_joint_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_kfe_joint_pub Publisher Connected")

        rospy.logdebug("All Joint Publishers READY")

    def joint_mono_des_callback(self, msg):
        rospy.logdebug(str(msg.joint_state.position))

        self.move_joints(msg.joint_state.position)

    def move_hand(self, current_position, next_position):
        """
        Move joints angle by controller of action in ROS.
        Action is unsynchronous communication.
        :param array current: positions of current joint
        :param array next_positions: positions of next joint
        :return
        """
        
        # Set current point (type is JointTrajectoryPoint)
        current_point = JointTrajectoryPoint()
        current_point.positions = current_positions
        current_point.time_from_start = rospy.Duration(1.0)

        # Set next point (type is JointTrajectoryPoint)
        next_point = JointTrajectoryPoint()
        next_point.positions = next_positions
        next_point.time_from_start = rospy.Duration(2.0)
        
        # Set goal
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory.joint_names = self.joint_names
        self.goal.trajectory.points = [current_point, next_point]
        self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)

        self._right_arm_action_client.send_goal(self.goal)
        rospy.logdebug("Success" if self._right_arm_action_client.wait_for_result() else "Failed")

if __name__=="__main__":
    rospy.init_node('joint_publisher_node', log_level=rospy.WARN)
    joint_publisher = JointPub()
    rate_value = 8.0
    #joint_publisher.start_loop(rate_value)
    #joint_publisher.start_sinus_loop(rate_value)
