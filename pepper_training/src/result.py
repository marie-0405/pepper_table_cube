# coding: UTF-8
import matplotlib.pyplot as plt
import pandas as pd
import rospkg
import rospy
import pandas


class Result(object):

  def __init__(self, rewards, succeeds, q_matrix):
    self.q_matrix = q_matrix
    self.rewards = rewards
    self.succeeds = succeeds
    self.df = pd.DataFrame({"reward": rewards, "succeeds": succeeds})
