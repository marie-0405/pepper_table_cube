#!/usr/bin/env python

import rospy
import copy
from gazebo_msgs.msg import ContactsState, ModelStates, LinkStates
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
import tf
import numpy
import math

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

    def __init__(self, min_distance, max_distance, list_of_observations, joint_limits, episode_done_criteria, joint_increment_value = 0.05, done_reward = -1000.0, alive_reward=10.0, weight_r1=1.0, weight_r2=1.0, discrete_division=10, maximum_base_linear_acceleration=3000.0, maximum_base_angular_velocity=20.0, maximum_joint_effort=10.0):
        rospy.logdebug("Starting pepperState Class object...")
        self.desired_length = Vector3(0.0, 0.0, 0.0)
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._alive_reward = alive_reward

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
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/pepper_dcm/joint_states", JointState, self.joints_state_callback)

        # We use it to get the positions of models.
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)

        # We use it to get the position of hand.
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback)
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
                rospy.logdebug("Current joint_states READY")
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

    def get_model_position(self, model_name):
        index = self.model_states.name.index(model_name)
        return self.model_states.pose[index].position

    def get_link_position(self, link_name):
        index = self.link_states.name.index(link_name)
        return self.link_states.pose[index].position

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
    
    def joints_state_callback(self,msg):
        self.joints_state = msg

    def model_states_callback(self, msg):
        self.model_states = msg

    def link_states_callback(self, msg):
        self.link_states = msg

    def pepper_height_ok(self):

        height_ok = self._min_height <= self.get_base_height() < self._max_height
        return height_ok

    def pepper_orientation_ok(self):

        orientation_rpy = self.get_base_rpy()
        roll_ok = self._abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self._abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok

    def calculate_reward_joint_position(self, weight=1.0):
        """
        We calculate reward base on the joints configuration. The more near 0 the better.
        :return:
        """
        acumulated_joint_pos = 0.0
        for joint_pos in self.joints_state.position:
            # Abs to remove sign influence, it doesnt matter the direction of turn.
            acumulated_joint_pos += abs(joint_pos)
            rospy.logdebug("calculate_reward_joint_position>>acumulated_joint_pos=" + str(acumulated_joint_pos))
        reward = weight * acumulated_joint_pos
        rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
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
            rospy.logdebug("calculate_reward_joint_effort>>joint_effort=" + str(joint_effort))
            rospy.logdebug("calculate_reward_joint_effort>>acumulated_joint_effort=" + str(acumulated_joint_effort))
        reward = weight * acumulated_joint_effort
        rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(reward))
        return reward

    def calculate_reward_contact_force(self, weight=1.0):
        """
        We calculate reward base on the contact force.
        The nearest to the desired contact force the better.
        We use exponential to magnify big departures from the desired force.
        Default ( 7.08 N ) desired force was taken from reading of the robot touching
        the ground from a negligible height of 5cm.
        :return:
        """
        force_magnitude = self.get_contact_force_magnitude()
        force_displacement = force_magnitude - self._desired_force

        rospy.logdebug("calculate_reward_contact_force>>force_magnitude=" + str(force_magnitude))
        rospy.logdebug("calculate_reward_contact_force>>force_displacement=" + str(force_displacement))
        # Abs to remove sign
        reward = weight * abs(force_displacement)
        rospy.logdebug("calculate_reward_contact_force>>reward=" + str(reward))
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
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_total_reward(self):
        """
        We consider VERY BAD REWARD -7 or less
        Perfect reward is 0.0, and total reward 1.0.
        The defaults values are chosen so that when the robot has fallen or very extreme joint config:
        r1 = -8.04
        r2 = -8.84
        r3 = -7.08
        r4 = -10.0 ==> We give priority to this, giving it higher value.
        :return:
        """

        r1 = self.calculate_reward_joint_position(self._weight_r1)
        r2 = self.calculate_reward_joint_effort(self._weight_r2)

        # The sign depend on its function.
        total_reward = self._alive_reward - r1 - r2

        rospy.logdebug("###############")
        rospy.logdebug("alive_bonus=" + str(self._alive_reward))
        rospy.logdebug("r1 joint_position=" + str(r1))
        rospy.logdebug("r2 joint_effort=" + str(r2))
        rospy.logdebug("total_reward=" + str(total_reward))
        rospy.logdebug("###############")

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
        rospy.logdebug("List of Observations==>"+str(self._list_of_observations))
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

        rospy.logdebug(str(state_discrete))
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
                delta = self._max_distance - self._min_distance
                max_value = delta
                min_value = -delta
            elif obs_name == "distance_from_cube_to_target":
                delta = self._max_distance - self._min_distance
                max_value = delta
                min_value = -delta
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

            self._obs_range_dict[obs_name] = [min_value,max_value]

    def create_bins(self):
        """
        We create the Bins for the discretization of the observations
        self.desired_world_point = Vector3(0.0, 0.0, 0.0)
        self._min_height = min_height
        self._max_height = max_height
        self._abs_max_roll = abs_max_roll
        self._abs_max_pitch = abs_max_pitch
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._alive_reward = alive_reward
        self._desired_force = desired_force
        self._desired_yaw = desired_yaw


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
        # self.init_knee_value = copy.deepcopy(self.current_joint_pose[2])
        return self.current_joint_pose

    def get_action_to_position(self, action):
        """
        Here we have the Actions number to real joint movement correspondance.
        :param action: Integer that goes from 0 to 9, because we have 10 actions.
        :return:
        """

        rospy.logdebug("current joint pose>>>"+str(self.current_joint_pose))
        rospy.logdebug("Action Number>>>"+str(action))

        if action == 0: #Increment RShoulderPitch
            rospy.logdebug("Action Decided:Increment RShoulderPitch>>>")
            self.current_joint_pose[0] += self._joint_increment_value
        elif action == 1: #Decrement RShoulderPitch
            rospy.logdebug("Action Decided:Decrement RShoulderPitch>>>")
            self.current_joint_pose[0] -= self._joint_increment_value
        elif action == 2: #Increment RShoulderRoll
            rospy.logdebug("Action Decided:Increment RShoulderRoll>>>")
            self.current_joint_pose[1] += self._joint_increment_value
        elif action == 3: #Decrement RShoulderRoll
            rospy.logdebug("Action Decided:Decrement RShoulderRoll>>>")
            self.current_joint_pose[1] -= self._joint_increment_value
        elif action == 4: #Increment RElbowRoll
            rospy.logdebug("Action Decided:Increment RElbowRoll>>>")
            self.current_joint_pose[2] += self._joint_increment_value
        elif action == 5: #Decrement RElbowRoll
            rospy.logdebug("Action Decided:Decrement RElbowRoll>>>")
            self.current_joint_pose[2] -= self._joint_increment_value
        elif action == 6: #Increment RElbowYaw
            rospy.logdebug("Action Decided:Increment RElbowYaw>>>")
            self.current_joint_pose[3] += self._joint_increment_value
        elif action == 7: #Decrement RElbowYaw
            rospy.logdebug("Action Decided:Decrement RElbowYaw>>>")
            self.current_joint_pose[3] -= self._joint_increment_value
        elif action == 8: #Increment RWristYaw
            rospy.logdebug("Action Decided:Increment RWristYaw>>>")
            self.current_joint_pose[4] += self._joint_increment_value
        elif action == 9: #Decrement RWristYaw
            rospy.logdebug("Action Decided:Decrement RWristYaw>>>")
            self.current_joint_pose[4] -= self._joint_increment_value

        rospy.logdebug("action to move joint states>>>" + str(self.current_joint_pose))

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
        ( it fell basically )
        :return: reward, done
        """

        if "pepper_minimum_height" in self._episode_done_criteria:
            pepper_height_ok = self.pepper_height_ok()
        else:
            rospy.logdebug("pepper_height_ok NOT TAKEN INTO ACCOUNT")
            pepper_height_ok = True

        if "pepper_vertical_orientation" in self._episode_done_criteria:
            pepper_orientation_ok = self.pepper_orientation_ok()
        else:
            rospy.logdebug("pepper_orientation_ok NOT TAKEN INTO ACCOUNT")
            pepper_orientation_ok = True

        rospy.logdebug("pepper_height_ok="+str(pepper_height_ok))
        rospy.logdebug("pepper_orientation_ok=" + str(pepper_orientation_ok))

        done = not(pepper_height_ok and pepper_orientation_ok)
        if done:
            rospy.logerr("It fell, so the reward has to be very low")
            total_reward = self._done_reward
        else:
            rospy.logdebug("Calculate normal reward because it didn't fall.")
            total_reward = self.calculate_total_reward()

        return total_reward, done

    def testing_loop(self):

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.calculate_total_reward()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('pepper_state_node', anonymous=True, log_level=rospy.DEBUG)
    max_height = 3.0
    min_height = 0.5
    max_incl = 1.57
    joint_increment_value = 0.32
    list_of_observations = ["base_roll",
                            "base_pitch",
                            "base_angular_vel_x",
                            "base_angular_vel_y",
                            "base_angular_vel_z",
                            "base_linear_acceleration_x",
                            "base_linear_acceleration_y",
                            "base_linear_acceleration_z"]
    joint_limits = {"haa_max": 1.6,
                     "haa_min": -1.6,
                     "hfe_max": 1.6,
                     "hfe_min": -1.6,
                     "kfe_max": 0.0,
                     "kfe_min": -1.6
                     }
    episode_done_criteria = [ "pepper_minimum_height",
                              "pepper_vertical_orientation"]
    done_reward = -1000.0
    alive_reward = 100.0
    desired_force = 7.08
    desired_yaw = 0.0
    weight_r1 = 0.0 # Weight for joint positions ( joints in the zero is perfect )
    weight_r2 = 0.0 # Weight for joint efforts ( no efforts is perfect )
    weight_r3 = 0.0 # Weight for contact force similar to desired ( weight of pepper )
    weight_r4 = 10.0 # Weight for orientation ( vertical is perfect )
    weight_r5 = 10.0 # Weight for distance from desired point ( on the point is perfect )
    discrete_division = 10
    maximum_base_linear_acceleration = 3000.0
    pepper_state = pepperState(   max_height=max_height,
                                    min_height=min_height,
                                    abs_max_roll=max_incl,
                                    abs_max_pitch=max_incl,
                                    joint_increment_value=joint_increment_value,
                                    list_of_observations=list_of_observations,
                                    joint_limits=joint_limits,
                                    episode_done_criteria=episode_done_criteria,
                                    done_reward=done_reward,
                                    alive_reward=alive_reward,
                                    desired_force=desired_force,
                                    desired_yaw=desired_yaw,
                                    weight_r1=weight_r1,
                                    weight_r2=weight_r2,
                                    weight_r3=weight_r3,
                                    weight_r4=weight_r4,
                                    weight_r5=weight_r5,
                                    discrete_division=discrete_division,
                                    maximum_base_linear_acceleration=maximum_base_linear_acceleration
                                                )
    pepper_state.testing_loop()
