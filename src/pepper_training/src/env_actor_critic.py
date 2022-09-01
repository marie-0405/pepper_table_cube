#!/usr/bin/env python
# coding: UTF-8


from functools import reduce
import gym
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
from result_controller import ResultController

from pepper_env_joint import PepperEnvJoint


if __name__ == '__main__':
    
  rospy.init_node('pepper_gym', anonymous=True, log_level=rospy.INFO)

  # Create the Gym environment
  env = gym.make('Pepper-v0')  # TODO change
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
      # Initialises the algorithm that we are going to use for learning
      ql = qlearn.QLearn(env=env, actions=range(env.action_space.n),
                          alpha=Alpha, gamma=Gamma, epsilon=Epsilon, 
                          eps_begin=eps_begin, eps_end=eps_end, nsteps=nsteps)
      initial_epsilon = ql.epsilon

      # Initializes the information of learning
      start_time = time.time()
      highest_reward = 0
      rewards = []
      succeeds = []
      # Starts the main training loop: the one about the episodes to do
      for episode in range(nepisodes):
          rospy.loginfo ("STARTING Episode #" + str(episode))
          
          cumulated_reward = 0
          cumulated_reward_msg = Float64()
          episode_reward_msg = Float64()
          done = False
          max_step = False

          # if qlearn.epsilon > 0.05:
          #     qlearn.epsilon *= epsilon_discount
          
          # Initialize the environment and get first state of the robot
          rospy.logdebug("env.reset...")
          # Now We return directly the stringuified observations called state
          state = env.reset()

          rospy.logdebug("env.get_state...==>" + str(state))
          
          # for each episode, we test the robot for nsteps
          for i in range(nsteps):

              # Pick an action based on the current state
              action = ql.chooseAction(state, i)
              # Execute the action in the environment and get feedback
              rospy.loginfo("###################### Start Step...["+str(i)+"]")
              rospy.loginfo("Epsilon" + str(ql.exp_strat.epsilon))
              # rospy.logdebug("RSP+,RSP-,RSR+,RSR-,RER+,RER-,REY+,REY-,RWY+,RWY- >> [0,1,2,3,4,5,6,7,8,9]")
              rospy.logdebug("Action to Perform >> "+str(action))
              nextState, reward, done, info = env.step(action)
              rospy.loginfo("Reward ==> " + str(reward))
              cumulated_reward += reward
              if highest_reward < cumulated_reward:
                  highest_reward = cumulated_reward

              rospy.logdebug("env.get_state...[distance_from_cube_to_target,distance from hand to cube]==>" + str(nextState))

              # Make the algorithm learn based on the results
              ql.learn(state, action, reward, nextState)
              q_matrix = ql.get_Q_matrix()
              rospy.logdebug(q_matrix)

              # We publish the cumulated reward
              cumulated_reward_msg.data = cumulated_reward
              reward_pub.publish(cumulated_reward_msg)
              if i == nsteps - 1:
                  max_step = True

              # 終了判定（タスク成功 or 最大ステップ）
              if not(done) and not(max_step):
                  state = nextState
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
          episode_reward_msg.data = cumulated_reward
          episode_reward_pub.publish(episode_reward_msg)
          rospy.loginfo("rewards: " + str(rewards))
          rospy.loginfo( ("EP: "+str(episode+1)+" - [alpha: "+str(round(ql.alpha,2))+" - gamma: "+str(round(ql.gamma,2))+" - epsilon: "+str(round(ql.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

      rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(ql.alpha)+"|"+str(ql.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

      l = last_time_steps.tolist()
      l.sort()

      rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
      rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

      ## Save the information of results
      result_controller = ResultController("a={}-g={}".format(Alpha, Gamma))
      result_controller.write(rewards, succeeds, q_matrix)
      result_controller.plot_reward() 
    
    env.close()
