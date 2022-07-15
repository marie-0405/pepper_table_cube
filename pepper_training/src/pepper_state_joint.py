#!/usr/bin/env python
# coding: UTF-8

import rospy
import copy
from gazebo_msgs.msg import ContactsState, ModelStates, LinkStates
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import tf
import numpy
import math

import time_recorder

"""
 wrenches:
      -
        force:
          x: -0.134995398774
          y: -0.252811705608
          z: -0.0861598399337
        torque:
          x: -0.00194729925705
          y: 0.028723398244
          z: -0.081229664152
    total_wrench:
      force:
        x: -0.134995398774
        y: -0.252811705608
        z: -0.0861598399337
      torque:
        x: -0.00194729925705
        y: 0.028723398244
        z: -0.081229664152
    contact_positions:
      -
        x: -0.0214808318267
        y: 0.00291348151391
        z: -0.000138379966267
    contact_normals:
      -
        x: 0.0
        y: 0.0
        z: 1.0
    depths: [0.000138379966266991]
  -
    info: "Debug:  i:(2/4)     my geom:pepper::lowerleg::lowerleg_contactsensor_link_collision_1\
  \   other geom:ground_plane::link::collision         time:50.405000000\n"
    collision1_name: "pepper::lowerleg::lowerleg_contactsensor_link_collision_1"
    collision2_name: "ground_plane::link::collision"

"""
"""
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
gazebo_msgs/ContactState[] states
  string info
  string collision1_name
  string collision2_name
  geometry_msgs/Wrench[] wrenches
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Wrench total_wrench
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Vector3[] contact_positions
    float64 x
    float64 y
    float64 z
  geometry_msgs/Vector3[] contact_normals
    float64 x
    float64 y
    float64 z
  float64[] depths
"""

class PepperState(object):

    def __init__(self, min_distance, max_distance, max_simulation_time, list_of_observations, joint_limits, episode_done_criteria, joint_increment_value = 0.05, done_reward = -1000.0, base_reward=10.0, success_reward=1000.0, weight_r1=1.0, weight_r2=1.0, discrete_division=10, maximum_base_linear_acceleration=3000.0, maximum_base_angular_velocity=20.0, maximum_joint_effort=10.0):
        rospy.logdebug("Starting pepperState Class object...")
        self.desired_length = Vector3(0.0, 0.0, 0.0)
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._max_simulation_time = max_simulation_time
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._base_reward = base_reward
        self._success_reward = success_reward

        self._weight_r1 = weight_r1
        self._weight_r2 = weight_r2

        self._list_of_observations = list_of_observations

        # Dictionary with the max and min of each of the joints
        self._joint_limits = joint_limits

        # Maximum base linear acceleration values
        self.maximum_base_linear_acceleration = maximum_base_linear_acceleration

        # Maximum Angular Velocity value
        # By maximum means the value that we consider relevant maximum, the sensor might pick up higher
        # But its an equilibrium between precission and number of divisions of the sensors data.
        self.maximum_base_angular_velocity = maximum_base_angular_velocity

        self.maximum_joint_effort = maximum_joint_effort

        # List of all the Done Episode Criteria
        self._episode_done_criteria = episode_done_criteria
        assert len(self._episode_done_criteria) != 0, "Episode_done_criteria list is empty. Minimum one value"

        self._discrete_division = discrete_division

        # We init the observation ranges and We create the bins now for all the observations
        self.init_bins()

        self.base_position = Point()
        self.base_orientation = Quaternion()
        self.base_angular_velocity = Vector3()
        self.base_linear_acceleration = Vector3()
        self.joints_state = JointState()

        # Odom we only use it for the height detection and planar position ,
        #  because in real robots this data is not trivial.
        # rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # We use the simulation_time to decide learning has been done.
        rospy.Subscriber("/clock", Clock, self.simulation_time_callback)
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/pepper_dcm/joint_states", JointState, self.joints_state_callback)
        # We use it to get the positions of models.
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.models_state_callback)
        # We use it to get the position of hand.
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.links_state_callback)
    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
 
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("pepper_dcm/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                # rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def set_desired_length(self, x, y, z):
        """
        Point where you want the cube to be from target
        :return:
        """
        self.desired_length.x = x
        self.desired_length.y = y
        self.desired_length.z = z
    
    def get_simulation_time_in_secs(self):
        return self.simulation_time.clock.secs

    def get_model_position(self, model_name):
        index = self.models_state.name.index(model_name)
        return self.models_state.pose[index].position

    def get_link_position(self, link_name):
        index = self.links_state.name.index(link_name)
        return self.links_state.pose[index].position

    def get_distance_from_point_to_point(self, p_from, p_to):
        """
        Given a Vector3 Object, get distance from current position
        
        Parameters
        ---------- 
        p_from: Vector3
            position of reference point
        p_to: Vector3
            position of target point

        Returns
        -------
        distance: int
            distance from reference point to target point
        """
        a = numpy.array((p_from.x, p_from.y, p_from.z))
        b = numpy.array((p_to.x, p_to.y, p_to.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_joint_states(self):
        return self.joints_state
    
    def get_joint_positions(self, joint_names):
        positions = []
        for joint_name in joint_names:
            index = self.joints_state.name.index(joint_name)
            positions.append(self.joints_state.position[index])
        return positions
    
    def simulation_time_callback(self, msg):
        self.simulation_time = msg

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def models_state_callback(self, msg):
        self.models_state = msg

    def links_state_callback(self, msg):
        self.links_state = msg

    def simulation_time_ok(self):
        simulation_time_ok = self._max_simulation_time >= self.get_simulation_time_in_secs()
        return simulation_time_ok
    
    def task_ok(self):
        """
        The task is that the cube is moved to the target position. 
        If the distance from cube to target is smaller than 0.03m, 
        task is done.
        """
        cube_pos = self.get_model_position("cube")
        target_pos = self.get_model_position("target")
        distance_from_cube_to_target = \
            self.get_distance_from_point_to_point(cube_pos, target_pos)
        task_ok = distance_from_cube_to_target <= 0.03
        return task_ok

    def calculate_reward_distance(self, weight, p_from, p_to):
        """
        We calculate reward base on the position between hand and cube. The more near 0 the better.
        
        Parameters
        ---------- 
        weight: int
            weight of reward
        p_from: Vector3
            position of reference point
        p_to: Vector3
            position of target point
        
        Returns
        ---------- 
        reward: int
        """
        distance = self.get_distance_from_point_to_point(p_from, p_to)
        rospy.loginfo("Distance" + str(distance))
        reward = weight * (distance - 0.025)
        return reward

    def calculate_reward_joint_position(self, weight=1.0):
        """
        We calculate reward base on the position between hand and cube. The more near 0 the better.
        :return:
        """
        acumulated_joint_pos = 0.0
        for joint_pos in self.joints_state.position:
            # Abs to remove sign influence, it doesnt matter the direction of turn.
            acumulated_joint_pos += abs(joint_pos)
            # rospy.logdebug("calculate_reward_joint_position>>acumulated_joint_pos=" + str(acumulated_joint_pos))
        reward = weight * acumulated_joint_pos
        # rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
        return reward

    def calculate_reward_joint_effort(self, weight=1.0):
        """
        We calculate reward base on the joints effort readings. The more near 0 the better.
        :return:
        """
        acumulated_joint_effort = 0.0
        for joint_effort in self.joints_state.effort:
            # Abs to remove sign influence, it doesnt matter the direction of the effort.
            acumulated_joint_effort += abs(joint_effort)
            # rospy.logdebug("calculate_reward_joint_effort>>joint_effort=" + str(joint_effort))
            # rospy.logdebug("calculate_reward_joint_effort>>acumulated_joint_effort=" + str(acumulated_joint_effort))
        reward = weight * acumulated_joint_effort
        # rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(reward))
        return reward

    def calculate_reward_orientation(self, weight=1.0):
        """
        We calculate the reward based on the orientation.
        The more its closser to 0 the better because it means its upright
        desired_yaw is the yaw that we want it to be.
        to praise it to have a certain orientation, here is where to set it.
        :return:
        """
        curren_orientation = self.get_base_rpy()
        yaw_displacement = curren_orientation.z - self._desired_yaw
        rospy.logdebug("calculate_reward_orientation>>[R,P,Y]=" + str(curren_orientation))
        acumulated_orientation_displacement = abs(curren_orientation.x) + abs(curren_orientation.y) + abs(yaw_displacement)
        reward = weight * acumulated_orientation_displacement
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_reward_distance_from_des_point(self, weight=1.0):
        """
        We calculate the distance from the desired point.
        The closser the better
        :param weight:
        :return:reward
        """
        distance = self.get_distance_from_point(self.desired_world_point)
        reward = weight * distance
        # rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_total_reward(self):
        """
        We consider VERY BAD REWARD -7 or less
        Perfect reward is 0.0, and total reward 1.0.
        The defaults values are chosen so that when the robot has fallen or very extreme joint config:
        r1 = -8.84 
        r2 = -10.0  ==> We give priority to this, giving it higher value.
        :return:
        """
        cube_pos = self.get_model_position("cube")
        target_pos = self.get_model_position("target")
        hand_pos = self.get_link_position("pepper::r_gripper")

        r1 = self.calculate_reward_distance(self._weight_r1, hand_pos, cube_pos)
        r2 = self.calculate_reward_distance(self._weight_r2, cube_pos, target_pos)

        # The sign depend on its function.
        total_reward = self._base_reward - r1 - r2

        rospy.logdebug("########################")
        rospy.logdebug("base_reward=" + str(self._base_reward))
        rospy.logdebug("r1 distance_from_hand_to_cube=" + str(r1))
        rospy.logdebug("r2 distance_from_cube_to_target=" + str(r2))
        rospy.logdebug("total_reward=" + str(total_reward))
        rospy.logdebug("#######################")

        return total_reward

    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array of the:
        1) distance from cube to target
        2) distance from hand to cube

        observation = [distance_from_cube_to_target,
                    - distance from hand to cube]

        :return: observation
        """
        
        cube_pos = self.get_model_position("cube")
        target_pos = self.get_model_position("target")
        hand_pos = self.get_link_position("pepper::r_gripper")

        distance_from_hand_to_cube = \
         self.get_distance_from_point_to_point(hand_pos, cube_pos)
        distance_from_cube_to_target = \
         self.get_distance_from_point_to_point(cube_pos, target_pos)

        observation = []
        # rospy.logdebug("List of Observations==>"+str(self._list_of_observations))
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_hand_to_cube":
                observation.append(distance_from_hand_to_cube)
            elif obs_name == "distance_from_cube_to_target":
                observation.append(distance_from_cube_to_target)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

        return observation

    def get_state_as_string(self, observation):
        """
        This function will do two things:
        1) It will make discrete the observations
        2) Will convert the discrete observations in to state tags strings
        :param observation:
        :return: state
        """
        observations_discrete = self.assign_bins(observation)
        string_state = ''.join(map(str, observations_discrete))
        rospy.logdebug("STATE==>"+str(string_state))
        return string_state

    def assign_bins(self, observation):
        """
        Will make observations discrete by placing each value into its corresponding bin
        :param observation:
        :return:
        """
        rospy.logdebug("Observations>>"+str(observation))

        state_discrete = numpy.zeros(len(self._list_of_observations), dtype=numpy.int32)
        for i in range(len(self._list_of_observations)):
            # We convert to int because anyway it will be round floats. We add Right True to include limits
            # Ex: [-20, 0, 20], value=-20 ==> index=0, In right = False, would be index=1
            state_discrete[i] = int(numpy.digitize(observation[i], self._bins[i], right=True))
            rospy.logdebug("bin="+str(self._bins[i])+"obs="+str(observation[i])+",end_val="+str(state_discrete[i]))

        # rospy.logdebug(str(state_discrete))
        return state_discrete

    def init_bins(self):
        """
        We initalize all related to the bins
        :return:
        """
        self.fill_observations_ranges()
        self.create_bins()

    def fill_observations_ranges(self):
        """
        We create the dictionary for the ranges of the data related to each observation
        :return:
        """
        self._obs_range_dict = {}
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_hand_to_cube":
                # We consider the range as based on the range of distance between models
                max_value = self._max_distance
                min_value = self._min_distance
            elif obs_name == "distance_from_cube_to_target":
                max_value = self._max_distance
                min_value = self._min_distance
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

            self._obs_range_dict[obs_name] = [min_value,max_value]

    def create_bins(self):
        """
        We create the Bins for the discretization of the observations
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._done_reward = done_reward
        self._base_reward = base_reward
        self._success_reward = success_reward

        :return:bins
        """
        number_of_observations = len(self._list_of_observations)
        parts_we_disrcetize = self._discrete_division
        rospy.logdebug("Parts to discretise==>"+str(parts_we_disrcetize))
        self._bins = numpy.zeros((number_of_observations, parts_we_disrcetize))
        for counter in range(number_of_observations):
            obs_name = self._list_of_observations[counter]
            min_value = self._obs_range_dict[obs_name][0]
            max_value = self._obs_range_dict[obs_name][1]
            self._bins[counter] = numpy.linspace(min_value, max_value, parts_we_disrcetize)

            rospy.logdebug("bins==>" + str(self._bins[counter]))

    def init_joints_pose(self, des_init_pos):
        """
        We initialise the Position variable that saves the desired length where we want our
        joints to be
        :param init_pos:
        :return:
        """
        self.current_joint_pose =[]
        self.current_joint_pose = copy.deepcopy(des_init_pos)
        return self.current_joint_pose
    
    # @time_recorder.time_recorder
    def get_action_to_position(self, action):
        """
        Here we have the Actions number to real joint movement correspondance.
        :param action: Integer that goes from 0 to 9, because we have 10 actions.
        :return:
        """

        rospy.logdebug("current joint pose>>>"+str(self.current_joint_pose))
        rospy.logdebug("Action Number>>>"+str(action))

        if action == 0: #Increment RShoulderPitch
            rospy.loginfo("Action Decided:Increment RShoulderPitch>>>")
            self.current_joint_pose[0] += self._joint_increment_value
        elif action == 1: #Decrement RShoulderPitch
            rospy.loginfo("Action Decided:Decrement RShoulderPitch>>>")
            self.current_joint_pose[0] -= self._joint_increment_value
        elif action == 2: #Increment RShoulderRoll
            rospy.loginfo("Action Decided:Increment RShoulderRoll>>>")
            self.current_joint_pose[1] += self._joint_increment_value
        elif action == 3: #Decrement RShoulderRoll
            rospy.loginfo("Action Decided:Decrement RShoulderRoll>>>")
            self.current_joint_pose[1] -= self._joint_increment_value
        elif action == 4: #Increment RElbowRoll
            rospy.loginfo("Action Decided:Increment RElbowRoll>>>")
            self.current_joint_pose[2] += self._joint_increment_value
        elif action == 5: #Decrement RElbowRoll
            rospy.loginfo("Action Decided:Decrement RElbowRoll>>>")
            self.current_joint_pose[2] -= self._joint_increment_value
        elif action == 6: #Increment RElbowYaw
            rospy.loginfo("Action Decided:Increment RElbowYaw>>>")
            self.current_joint_pose[3] += self._joint_increment_value
        elif action == 7: #Decrement RElbowYaw
            rospy.loginfo("Action Decided:Decrement RElbowYaw>>>")
            self.current_joint_pose[3] -= self._joint_increment_value
        elif action == 8: #Increment RWristYaw
            rospy.loginfo("Action Decided:Increment RWristYaw>>>")
            self.current_joint_pose[4] += self._joint_increment_value
        elif action == 9: #Decrement RWristYaw
            rospy.loginfo("Action Decided:Decrement RWristYaw>>>")
            self.current_joint_pose[4] -= self._joint_increment_value

        # rospy.logdebug("action to move joint states>>>" + str(self.current_joint_pose))

        self.clamp_to_joint_limits()

        return self.current_joint_pose

    def clamp_to_joint_limits(self):
        """
        clamps self.current_joint_pose based on the joint limits
        self._joint_limits
        {"r_shoulder_pitch_max": r_shoulder_pitch_max,
         "r_shoulder_pitch_min": r_shoulder_pitch_min,
         "r_shoulder_roll_max": r_shoulder_roll_max,
         "r_shoulder_roll_min": r_shoulder_roll_min,
         "r_elbow_roll_max": r_elbow_roll_max,
         "r_elbow_roll_min": r_elbow_roll_min,
         "r_elbow_yaw_max": r_elbow_yaw_max,
         "r_elbow_yaw_min": r_elbow_yaw_min,
         "r_wrist_yaw_max": r_wrist_yaw_max,
         "r_wrist_yaw_min": r_wrist_yaw_min,
        }
        :return:
        """

        rospy.logdebug("Clamping current_joint_pose>>>" + str(self.current_joint_pose))
        rsp_joint_value = self.current_joint_pose[0]
        rsr_joint_value = self.current_joint_pose[1]
        rer_joint_value = self.current_joint_pose[2]
        rey_joint_value = self.current_joint_pose[3]
        rwy_joint_value = self.current_joint_pose[4]

        rospy.logdebug("rsp_min>>>" + str(self._joint_limits["rsp_min"]))
        rospy.logdebug("rsp_max>>>" + str(self._joint_limits["rsp_max"]))
        self.current_joint_pose[0] = max(min(rsp_joint_value, self._joint_limits["rsp_max"]),
                                         self._joint_limits["rsp_min"])
        self.current_joint_pose[1] = max(min(rsr_joint_value, self._joint_limits["rsr_max"]),
                                         self._joint_limits["rsr_min"])
        self.current_joint_pose[2] = max(min(rer_joint_value, self._joint_limits["rer_max"]),
                                         self._joint_limits["rer_min"])
        self.current_joint_pose[3] = max(min(rey_joint_value, self._joint_limits["rey_max"]),
                                         self._joint_limits["rey_min"])
        self.current_joint_pose[4] = max(min(rwy_joint_value, self._joint_limits["rwy_max"]),
                                         self._joint_limits["rwy_min"])

        rospy.logdebug("DONE Clamping current_joint_pose>>>" + str(self.current_joint_pose))


    def process_data(self):
        """
        We return the total reward based on the state in which we are in and if its done or not
        :return: reward, done
        """
        if "cube_moved_target" in self._episode_done_criteria:
            task_ok = self.task_ok()
        else:
            rospy.logdebug("cube_moved_target NOT TAKEN INTO ACCOUNT")
            task_ok = True

        rospy.loginfo("task_ok="+str(task_ok))

        done = task_ok
        if done:
            rospy.logerr("The reward has to be very high because it is succeeded.")
            total_reward = self._success_reward
        else:
            rospy.logdebug("Calculate normal reward because it is continued.")
            total_reward = self.calculate_total_reward()

        return total_reward, done
    def testing_loop(self):

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.calculate_total_reward()
            rate.sleep()
            