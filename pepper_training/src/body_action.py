#!/usr/bin/env python
# coding: UTF-8

import actionlib
import rospy
import math
import time
import copy
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

import time_recorder

DURATION = 0.25


class BodyAction(object):
    def __init__(self, joint_names, action):
        self.joint_names = joint_names
        self._right_arm_action_client = actionlib.SimpleActionClient(action, FollowJointTrajectoryAction)
        self._right_arm_action_client.wait_for_server()

    def set_init_pose(self, init_pose):
        """
        Sets joints to initial position [0,0,0]
        :return: The init Pose
        """
        self.check_publishers_connection()
        self.move_joints(init_pose)

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

    @time_recorder.time_recorder
    def move_joints(self, current_positions, next_positions):
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
        current_point.time_from_start = rospy.Duration(0.20)

        # Set next point (type is JointTrajectoryPoint)
        next_point = JointTrajectoryPoint()
        next_point.positions = next_positions
        next_point.time_from_start = rospy.Duration(0.6)
        
        # Set goal
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory.joint_names = self.joint_names
        self.goal.trajectory.points = [current_point, next_point]
        self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.25)

        self._right_arm_action_client.send_goal(self.goal)
        rospy.loginfo("Success" if self._right_arm_action_client.wait_for_result() else "Failed")

    def move_joints_jump(self, joints_array, do_jump = False):

        if not do_jump:
            # We jump in the direction given
            self.prepare_jump(joints_array)
        else:
            # We Dont want to jump so we move the joint to the given position
            self.jump(joints_array)

    def jump(self, joints_array):
        """
        Simply Extends the Knee to maximum and gives time to perform jump until next call
        :param joints_array:
        :return:
        """

        kfe_final = 0.0
        wait_for_jump_time = 0.15

        #time.sleep(wait_for_jump_time*2)
        joints_array[2] = kfe_final
        rospy.logdebug("joints_array DEFLEX==>" + str(joints_array))
        self.move_joints(joints_array)
        # Wait for an amount of time
        rospy.logdebug("Performing Jump....")
        time.sleep(wait_for_jump_time)
        rospy.logdebug("Jump Done==>")


    def prepare_jump(self, joints_array):
        """
        Moves the joints and knee based on the desired joints_array.
        The knee value will be adjusted so that it always touches the
        foot before anything else, being aligned with the hip.
        joints_array = [haa,hfe,kfe]
        :param joints_array:
        :return:
        """
        # We need to deep copy otherwise it changed the parent object
        corrected_joint_array = copy.deepcopy(joints_array)


        # The flex value will be the given kneww joint value
        flex_value = abs(corrected_joint_array[2])
        l = 0.39587
        m = 0.38393
        hfe_flex = copy.deepcopy(corrected_joint_array[1]) + flex_value
        hfe_final = copy.deepcopy(corrected_joint_array[1])

        rospy.logdebug("################")
        rospy.logdebug("hfe_flex==>" + str(hfe_flex))
        rospy.logdebug("hfe_final==>" + str(hfe_final))

        # Alfa is the maximum value that hfe_flex can have to mantaining the tip of lower leg colineal with the
        # init end of the upper leg element. Beyond that, the knee would impact the floor,
        # before the lower leg would
        alfa = math.asin(m/l)
        rospy.logdebug("alfa_min<flex_value<alfa_max ? alfa=" + str(alfa) + " ?? flex_value=" + str(flex_value))
        # We clamp it to the maximum or minimum values
        flex_value = max(min(flex_value, alfa),-alfa)
        rospy.logdebug("CLAMPED hfe_flex==>" + str(flex_value))

        chi_value = l * math.sin(flex_value) / m
        assert abs(chi_value) <= 1.0, "Chi value impossible to calculate as acos"
        knee_angle = (math.pi/2.0) + flex_value - math.acos(chi_value)
        #knee_angle = -flex_value
        rospy.logdebug("knee_angle==>" + str(knee_angle))

        # Flex Knee
        # Its inverted the angle dir
        corrected_joint_array[1] = hfe_flex
        corrected_joint_array[2] = -abs(knee_angle)
        rospy.logdebug("joints_array FLEX==>" + str(corrected_joint_array))
        self.move_joints(corrected_joint_array)

        rospy.logdebug("Movement Done==>")


    def start_loop(self, rate_value = 2.0):
        rospy.logdebug("Start Loop")
        pos1 = [0.0,0.0,1.6]
        pos2 = [0.0,0.0,-1.6]
        position = "pos1"
        rate = rospy.Rate(rate_value)
        while not rospy.is_shutdown():
          if position == "pos1":
            self.move_joints(pos1)
            position = "pos2"
          else:
            self.move_joints(pos2)
            position = "pos1"
          rate.sleep()

    def start_sinus_loop(self, rate_value = 2.0):
        rospy.logdebug("Start Loop")
        w = 0.0
        x = 1.0*math.sin(w)
        #pos_x = [0.0,0.0,x]
        pos_x = [x, 0.0, 0.0]
        #pos_x = [0.0, x, 0.0]
        rate = rospy.Rate(rate_value)
        while not rospy.is_shutdown():
            self.move_joints(pos_x)
            w += 0.05
            x = 1.0 * math.sin(w)
            #pos_x = [0.0, 0.0, x]
            pos_x = [x, 0.0, 0.0]
            #pos_x = [0.0, x, 0.0]
            print x
            rate.sleep()

if __name__=="__main__":
    rospy.init_node('joint_publisher_node', log_level=rospy.WARN)
    joint_publisher = JointPub()
    rate_value = 8.0
    #joint_publisher.start_loop(rate_value)
    #joint_publisher.start_sinus_loop(rate_value)
