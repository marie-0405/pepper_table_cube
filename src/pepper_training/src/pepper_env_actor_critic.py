#!/usr/bin/env python
# coding: UTF-8

'''
    By Miguel Angel Rodriguez <duckfrost@theconstructsim.com>
    Visit our website at www.theconstructsim.com
'''
import gym
import rospy
import numpy as np
import random
import time
from gym import utils, spaces
from geometry_msgs.msg import Pose
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from body_action import BodyAction
from pepper_state_joint import PepperState
import time_recorder
from model_setter import ModelSetter

# register the training environment in the gym as an available one
reg = register(
    id='Pepper-v1',
    entry_point='pepper_env_actor_critic:PepperEnvActorCritic',
    )


class PepperEnvActorCritic(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment

        # gets training parameters from param server
        self.id = id
        self.desired_length = Pose()
        self.desired_length.position.x = rospy.get_param("/desired_length/x")
        self.desired_length.position.y = rospy.get_param("/desired_length/y")
        self.desired_length.position.z = rospy.get_param("/desired_length/z")
        self.running_step = rospy.get_param("/running_step")
        self.min_distance = rospy.get_param("/min_distance")
        self.max_distance = rospy.get_param("/max_distance")
        self.max_simulation_time = rospy.get_param("/max_simulation_time")
        self.joint_increment_value = rospy.get_param("/joint_increment_value")
        print("Joint increment value", self.joint_increment_value)
        self.done_reward = rospy.get_param("/done_reward")
        self.base_reward = rospy.get_param("/base_reward")
        self.success_reward = rospy.get_param("/success_reward")
        self.random_cube = rospy.get_param("/random_cube")
        self.random_cube_target = rospy.get_param("/random_cube_target")
        self.use_arms = rospy.get_param("/use_arms")

        self.list_of_observations = rospy.get_param("/list_of_observations")

        r_shoulder_pitch_max = rospy.get_param("/joint_limits_array/r_shoulder_pitch_max")
        r_shoulder_pitch_min = rospy.get_param("/joint_limits_array/r_shoulder_pitch_min")
        r_shoulder_roll_max = rospy.get_param("/joint_limits_array/r_shoulder_roll_max")
        r_shoulder_roll_min = rospy.get_param("/joint_limits_array/r_shoulder_roll_min")
        r_elbow_roll_max = rospy.get_param("/joint_limits_array/r_elbow_roll_max")
        r_elbow_roll_min = rospy.get_param("/joint_limits_array/r_elbow_roll_min")
        r_elbow_yaw_max = rospy.get_param("/joint_limits_array/r_elbow_yaw_max")
        r_elbow_yaw_min = rospy.get_param("/joint_limits_array/r_elbow_yaw_min")
        r_wrist_yaw_max = rospy.get_param("/joint_limits_array/r_wrist_yaw_max")
        r_wrist_yaw_min = rospy.get_param("/joint_limits_array/r_wrist_yaw_min")

        self.joint_limits = {"rsp_max": r_shoulder_pitch_max,
                             "rsp_min": r_shoulder_pitch_min,
                             "rsr_max": r_shoulder_roll_max,
                             "rsr_min": r_shoulder_roll_min,
                             "rer_max": r_elbow_roll_max,
                             "rer_min": r_elbow_roll_min,
                             "rey_max": r_elbow_yaw_max,
                             "rey_min": r_elbow_yaw_min,
                             "rwy_max": r_wrist_yaw_max,
                             "rwy_min": r_wrist_yaw_min,
                             }

        self.discrete_division = rospy.get_param("/discrete_division")  # ?

        self.maximum_base_linear_acceleration = rospy.get_param("/maximum_base_linear_acceleration")  # ?
        self.maximum_base_angular_velocity = rospy.get_param("/maximum_base_angular_velocity")  # ?
        self.maximum_joint_effort = rospy.get_param("/maximum_joint_effort")  # ?

        self.weight_r1 = rospy.get_param("/weight_r1")
        self.weight_r2 = rospy.get_param("/weight_r2")

        r_shoulder_pitch_init_value = rospy.get_param("/init_joint_pose/r_shoulder_pitch")
        r_shoulder_roll_init_value = rospy.get_param("/init_joint_pose/r_shoulder_roll")
        r_elbow_roll_init_value = rospy.get_param("/init_joint_pose/r_elbow_roll")
        r_elbow_yaw_init_value = rospy.get_param("/init_joint_pose/r_elbow_yaw")
        r_wrist_yaw_init_value = rospy.get_param("/init_joint_pose/r_wrist_yaw")


        l_shoulder_pitch_init_value = rospy.get_param("/init_joint_pose/l_shoulder_pitch")
        l_shoulder_roll_init_value = rospy.get_param("/init_joint_pose/l_shoulder_roll")
        l_elbow_roll_init_value = rospy.get_param("/init_joint_pose/l_elbow_roll")
        l_elbow_yaw_init_value = rospy.get_param("/init_joint_pose/l_elbow_yaw")
        l_wrist_yaw_init_value = rospy.get_param("/init_joint_pose/l_wrist_yaw")

        self.init_right_joint_pose = [r_shoulder_pitch_init_value,r_shoulder_roll_init_value,r_elbow_roll_init_value,
            r_elbow_yaw_init_value, r_wrist_yaw_init_value]
        self.init_left_joint_pose = [l_shoulder_pitch_init_value,l_shoulder_roll_init_value,l_elbow_roll_init_value,
            l_elbow_yaw_init_value, l_wrist_yaw_init_value]

        # Fill in the Done Episode Criteria list
        self.episode_done_criteria = rospy.get_param("/episode_done_criteria")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        self.controllers_object = ControllersConnection(namespace="pepper_dcm")

        self.pepper_state_object = PepperState(
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            max_simulation_time=self.max_simulation_time,
            list_of_observations=self.list_of_observations,
            joint_increment_value=self.joint_increment_value,
            joint_limits=self.joint_limits,
            episode_done_criteria=self.episode_done_criteria,
            done_reward=self.done_reward,
            base_reward=self.base_reward,
            success_reward=self.success_reward,
            weight_r1=self.weight_r1,
            weight_r2=self.weight_r2,
            discrete_division=self.discrete_division,
            maximum_base_linear_acceleration=self.maximum_base_linear_acceleration,
            maximum_base_angular_velocity=self.maximum_base_angular_velocity,
            maximum_joint_effort=self.maximum_joint_effort,
            object_name='cube'
        )
        self.right_arm_joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
        self.left_arm_joint_names = ["LShoulderPitch", "LShoulderRoll", "LElbowRoll", "LElbowYaw", "LWristYaw"]
        self.pepper_state_object.set_desired_length(self.desired_length.position.x,
                                                    self.desired_length.position.y,
                                                    self.desired_length.position.z)

        self.right_arm_action_object = BodyAction(self.right_arm_joint_names, 
            "/pepper_dcm/RightArm_controller/follow_joint_trajectory")
        self.left_arm_action_object = BodyAction(self.left_arm_joint_names, 
            "/pepper_dcm/LeftArm_controller/follow_joint_trajectory")        
        
        observation_num = len(self.list_of_observations)
        if self.use_arms:
            self.action_space = spaces.Discrete(20)
        else:
            self.action_space = spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=self.min_distance,
            high=self.max_distance,
            shape=(observation_num,)
        )
        self.reward_range = (-np.inf, np.inf)

        self.cube = ModelSetter("cube")
        self.target = ModelSetter("target")

        self._seed()  # TODO ランダムシードは固定しないほうがいいかも

    # A function to iline 49, in _initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    # @time_recorder.time_recorder
    def _reset(self):

        # 0st: We pause the Simulator
        rospy.logdebug("Pausing SIM...")
        self.gazebo.pauseSim()

        # 1st: resets the simulation to initial values
        rospy.logdebug("Reset SIM...")
        self.gazebo.resetSim()

        # 2nd: We Set the gravity to 0.0 so that we dont fall when reseting joints
        # It also UNPAUSES the simulation
        # rospy.logdebug("Remove Gravity...")
        self.gazebo.change_gravity(0.0, 0.0, 0.0)

        # EXTRA: Reset JoinStateControlers because sim reset doesnt reset TFs, generating time problems
        # rospy.logdebug("reset_pepper_joint_controllers...")
        self.controllers_object.reset_pepper_joint_controllers()

        if self.random_cube:
            # # EXTRA: move cube to random position    
            positions = [0.09, 0.11, 0.13, 0.15]  ## TODO delete 0.15 because arm cannot reach it
            rand_index = random.randint(0, 3)
            self.cube.set_position(positions[rand_index], -0.28, 0.73)
            rospy.loginfo("X init pose = " + str(positions[rand_index]))

        if self.random_cube_target:
            # # EXTRA: move cube and target to random position    
            cube_positions_x = [0.13, 0.14, 0.06]
            cube_positions_y = [-0.28, -0.18, -0.18] 
            target_positions_x = [0.12, 0.06, 0.15]
            target_positions_y = [-0.18, -0.18, -0.18]
            rand_index = random.randint(0, 2)
            self.cube.set_position(cube_positions_x[rand_index],cube_positions_y[rand_index] , 0.73)
            self.target.set_position(target_positions_x[rand_index],target_positions_y[rand_index] , 0.7055)
        
        if self.use_arms:
            cube_positions_x = [0.13, 0.13]  
            cube_positions_y = [-0.15, 0.15] 
            rand_index = random.randint(0, 1)
            self.cube.set_position(cube_positions_x[rand_index],cube_positions_y[rand_index] , 0.73)
            self.target.set_position(0.13, 0.0, 0.7055)

        # 4th: Check all scribers work.
        # Get the state of the Robot defined by its RPY orientation, distance from
        # desired point, contact force and JointState of the three joints
        rospy.logdebug("check_all_systems_ready...")
        self.pepper_state_object.check_all_systems_ready()

        self.pepper_state_object.set_init_distances()
        
        # 5th: We restore the gravity to original
        # rospy.logdebug("Restore Gravity...")
        self.gazebo.change_gravity(0.0, 0.0, -9.81)

        # 3rd: resets the robot to initial conditions
        # rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose))
        # We save that position as the current joint desired position
        # init_pos = self.pepper_state_object.init_joints_pose(self.init_joint_pose)
        current_right_position = self.pepper_state_object.get_joint_positions(self.right_arm_joint_names)
        current_left_position = self.pepper_state_object.get_joint_positions(self.left_arm_joint_names)
        # print("Current left position", current_left_position)
        # print('Init Position', self.init_joint_pose)
        self.right_arm_action_object.set_init_pose(current_right_position, self.init_right_joint_pose)
        if self.use_arms:
            self.left_arm_action_object.set_init_pose(current_left_position, self.init_left_joint_pose)


        # time.sleep(2)
        # 6th: pauses simulation
        rospy.logdebug("Pause SIM...")
        self.gazebo.pauseSim()

        # 7th: Get the State Discrete Stringuified version of the observations
        rospy.logdebug("get_observations...")
        observation = self.pepper_state_object.get_observations()
        state = observation

        return state

    # @time_recorder.time_recorder
    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot

        # 1st, decide which action corresponds to which position is incremented
        next_positions = self.pepper_state_object.get_action_to_position(self.right_arm_joint_names + self.left_arm_joint_names, action)

        # We move it to that pos
        self.gazebo.unpauseSim()
        ## Using Action
        # Get current positions of joint
        current_positions = self.pepper_state_object.get_joint_positions(self.right_arm_joint_names + self.left_arm_joint_names)
        # Then we send the command to the robot and let it go

        self.right_arm_action_object.move_joints(current_positions[:4], next_positions[:4])
        if self.use_arms:
            print('Current left position', current_positions[5:])
            print('Next position', next_positions[5:])
            self.left_arm_action_object.move_joints(current_positions[5:], next_positions[5:])
    
        # for running_step seconds
        time.sleep(self.running_step)
        self.gazebo.pauseSim()

        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.pepper_state_object.get_observations()
        # finally we get an evaluation based on what happened in the sim
        reward,done = self.pepper_state_object.process_data()

        state = observation

        return state, reward, done, {}

    def _render(self, human, close):
        return 'aaa'
