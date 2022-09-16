import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Actor(nn.Module):
  """Actor(
    (linear1): Linear(in_features=4, out_features=128, bias=True)
    (linear2): Linear(in_features=128, out_features=256, bias=True)
    (linear3): Linear(in_features=256, out_features=2, bias=True)
  )
  """
  def __init__(self, state_size, action_size, L1, L2):
    super(Actor, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.linear1 = nn.Linear(self.state_size, L1)
    self.linear2 = nn.Linear(L1, L2)
    self.linear3 = nn.Linear(L2, self.action_size)

  def forward(self, state):
    output = F.relu(self.linear1(state))
    output = F.relu(self.linear2(output))
    output = self.linear3(output)
    distribution = Categorical(F.softmax(output, dim=-1))
    return distribution
