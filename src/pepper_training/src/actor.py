import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Actor(nn.Module):
  # TODO 層の書き方をPytorchのチュートリアルと同じように、全てinitにまとめても同じかどうか確認して、同じならそうしたい。
  # modelの出力は、以下のようだった。なんで、Linearだけなのか？ReLuは入れてはいけないのか？
  """Actor(
    (linear1): Linear(in_features=4, out_features=128, bias=True)
    (linear2): Linear(in_features=128, out_features=256, bias=True)
    (linear3): Linear(in_features=256, out_features=2, bias=True)
  )
  """
  def __init__(self, state_size, action_size):
    super(Actor, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    # self.flatten = nn.Flatten()
    # self.linear_relu_stack = nn.Sequential(
    #   # ここでニューラルネットワークを作成している
    #   ## 入力が、状態数で、出力が128ベクトル
    #   nn.Linear(state_size, 128),
    #   nn.ReLU(),
    #   ## 入力が128ベクトルで出力が256ベクトル
    #   nn.Linear(128, 256),
    #   nn.ReLU(),
    #   ## 入力が256ベクトルで、出力が行動数
    #   nn.Linear(256, action_size),
    # )
    # ここでニューラルネットワークを作成している
    # 入力が、状態数で、出力が128ベクトル
    self.linear1 = nn.Linear(self.state_size, 128)
    
    # 入力が128ベクトルで出力が256ベクトル
    self.linear2 = nn.Linear(128, 256)
    # 入力が256ベクトルで、出力が行動数
    self.linear3 = nn.Linear(256, self.action_size)
  def forward(self, state):  # forward関数は、モデル（Actor）の使用時に他の操作とともに自動で呼び出される
    output = F.relu(self.linear1(state))
    output = F.relu(self.linear2(output))
    output = self.linear3(output)
    distribution = Categorical(F.softmax(output, dim=-1))
    
    # state = self.flatten(state)
    # logits = self.linear_relu_stack(state)
    # return logits
    """
    distributionのsizeは、2であり、右に動かすか左に動かすかの確率が格納されている
    ex)
    >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    >>> m.sample()  # equal probability of 0, 1, 2, 3
    tensor(3)
    """
    return distribution
