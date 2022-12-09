import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


class DropoutCritic(nn.Module):
  def __init__(self, state_size, action_size, L1, L2):
    super(DropoutCritic, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.linear1 = nn.Linear(self.state_size, L1)
    self.linear2 = nn.Linear(L1, L2)
    self.linear3 = nn.Linear(L2, 1)
    self.dropout = nn.Dropout(p=settings.dropout_rate)  ## TODO change

  def forward(self, state):
    output = F.relu(self.linear1(state))
    output = self.dropout(F.relu(self.linear2(output)))
    value = self.dropout(self.linear3(output))
    return value
