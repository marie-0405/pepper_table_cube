#!/usr/bin/env python
# coding: UTF-8


from functools import reduce
import gym
import json
import nep
import numpy as np
import qlearn
import random
import sys
import time
from std_msgs.msg import Float64

# ROS packages required
import rospy
import rospkg

# import my tool
from hyper_parameter import HyperParameter

# import my training environment
# import pepper_env_actor_critic
import pepper_env_joint



if __name__ == '__main__':

  rospy.init_node('pepper_gym', anonymous=True, log_level=rospy.INFO)

  node = nep.node("Environment")    # Create a new nep node
  conf = node.hybrid('192.168.11.62')
  # conf = node.hybrid('192.168.3.14')
  sub = node.new_sub("calc","json", conf)      # Set the topic and message type
  pub = node.new_pub("env", "json", conf)

  # define getting method for message
  def get_msg():
    while True:
      s, msg = sub.listen()
      if s:
        print(msg)
        return msg
      else:
        time.sleep(.0001)

  # Create the Gym environment
  env = gym.make('Pepper-v1')  # TODO

  # Get space from environment and publish it
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  msg = {'action_size': action_size, 'state_size': state_size}
  pub.publish(msg)
  lr = 0.0001  # 学習率

  rospy.logdebug ( "Gym environment done")
  reward_pub = rospy.Publisher('/pepper/reward', Float64, queue_size=1)
  episode_reward_pub = rospy.Publisher('/pepper/episode_reward', Float64, queue_size=1)

  # Set the logging system
  rospack = rospkg.RosPack()
  pkg_path = rospack.get_path('pepper_training')
  outdir = pkg_path + '/training_results'
  # env = wrappers.Monitor(env, outdir, force=True)
  # rospy.logdebug("Monitor Wrapper started")

  last_time_steps = np.ndarray(0)

  # Loads parameters from the ROS param server
  # Parameters are stored in a yaml file inside the config directory
  # They are loaded at runtime by the launch file
  Alpha = rospy.get_param("/alpha")
  Epsilon = rospy.get_param("/epsilon")
  eps_begin = rospy.get_param("/epsilon_begin")
  eps_end = rospy.get_param("/epsilon_end")
  Gamma = rospy.get_param("/gamma")
  epsilon_discount = rospy.get_param("/epsilon_discount")
  nepisodes = rospy.get_param("/nepisodes")
  nsteps = rospy.get_param("/nsteps")

  # If you want to do grid search, try this code on.
  # Alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  # Gammas = [0.9, 1.0]

  Alphas = [0.5]
  Gammas = [0.8]
  rospy.loginfo("Alphas ==> " + str(Alphas))
  rospy.loginfo("Gammas ==> " + str(Gammas))

  for g in Gammas:
    Gamma = g
    for a in Alphas:
      Alpha = a

      # Initializes the information of learning
      start_time = time.time()
      highest_reward = 0
      rewards = []
      succeeds = []
      # Starts the main training loop: the one about the episodes to do
      for episode in range(nepisodes):
        rospy.loginfo ("STARTING Episode #" + str(episode))
        cumulated_reward = 0
        # cumulated_reward_msg = Float64()
        # episode_reward_msg = Float64()
        done = False
        max_step = False

        # Initialize the environment and get first state of the robot
        rospy.loginfo("env.reset...")
        # Now We return directly the observations called state
        state = env.reset()
        pub.publish({'state': state})

        rospy.logdebug("env.get_state => " + str(state))

        # for each episode, we test the robot for nsteps
        for i in range(nsteps):

          # Get an action based on the current state from Actor's distribution
          action = np.array(get_msg()['action'])
          # Execute the action in the environment and get feedback
          rospy.loginfo("###################### Start Step...["+str(i)+"]")
          # rospy.logdebug("RSP+,RSP-,RSR+,RSR-,RER+,RER-,REY+,REY-,RWY+,RWY- >> [0,1,2,3,4,5,6,7,8,9]")
          rospy.loginfo("Action to Perform >> "+str(action))
          next_state, reward, done, info = env.step(action)
          pub.publish({'done': done, 'info': info, 'next_state': next_state, 'reward': reward})
          rospy.loginfo("Reward => " + str(reward))
          rospy.loginfo('Done => ' + str(done))
          cumulated_reward += reward
          if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

          rospy.loginfo("env.get_state => " + str(next_state))

          if i == nsteps - 1:
            max_step = True

          # 終了判定（タスク成功 or 最大ステップ）
          if not(done) and not(max_step):
            state = next_state
          else:
            last_time_steps = np.append(last_time_steps, [int(i + 1)])
            rospy.logdebug ("DONE")
            if max_step:
              succeeds.append(False)
            else:
              succeeds.append(True)
            break
          rospy.loginfo("###################### END Step...["+str(i)+"]")

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rewards.append(cumulated_reward)
        rospy.loginfo("rewards: " + str(rewards))
        rospy.loginfo( ("EP: "+str(episode+1)+" - [alpha: "+str(round(Alpha,2))+" - gamma: "+str(round(Gamma,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

      rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(Alpha)+"|"+str(Gamma)+"|"+str(highest_reward)+"| PICTURE |"))

      l = last_time_steps.tolist()
      l.sort()

      rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
      rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
  env.close()
