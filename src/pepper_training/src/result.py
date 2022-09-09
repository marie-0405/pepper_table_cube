# coding: UTF-8
import matplotlib.pyplot as plt
import pandas as pd


class Result(object):

  def __init__(self, rewards, succeeds, experiences, q_matrix, actor_losses, critic_losses):
    self.q_matrix = q_matrix
    self.rewards = rewards
    self.succeeds = succeeds
    self.experiences = experiences
    self.actor_losses = actor_losses
    self.critic_losses = critic_losses
    self.df = pd.DataFrame({"reward": rewards, "succeed": succeeds, "actor_loss": actor_losses, "critic_loss": critic_losses})
