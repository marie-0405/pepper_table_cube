# coding: UTF-8

import gym, os
from itertools import count
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from result_controller import ResultController


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001  # 学習率

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


class Critic(nn.Module):
  def __init__(self, state_size, action_size):
    super(Critic, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.linear1 = nn.Linear(self.state_size, 128)
    self.linear2 = nn.Linear(128, 256)
    self.linear3 = nn.Linear(256, 1)

  def forward(self, state):
    output = F.relu(self.linear1(state))
    output = F.relu(self.linear2(output))
    value = self.linear3(output)
    return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
  # 報酬値から収益を計算する
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def save_results(file_name_end, cumulated_rewards, succeeds, experiences='', actor_losses='', critic_losses=''):
  ## Save the results
  result_controller = ResultController(file_name_end)
  result_controller.write(cumulated_rewards,
                          succeeds=succeeds,
                          experiences=experiences,
                          actor_losses=actor_losses,
                          critic_losses=critic_losses)
  result_controller.plot('reward')
  result_controller.plot('actor_loss')
  result_controller.plot('critic_loss')

def trainIters(actor, critic, n_iters):
  # parameters()の中身はtensorがいくつかある
  optimizerA = optim.Adam(actor.parameters())
  optimizerC = optim.Adam(critic.parameters())
  cumulated_rewards = []
  succeeds = []
  actor_losses = []
  critic_losses = []
  cols = ['state', 'action', 'reward', 'next_state']
  experiences = pd.DataFrame(columns=cols)


  for iter in range(n_iters):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    env.reset()
    cumulated_rewards.append(0)

    for i in count():
      # 最適化されたすべての勾配を0にする（初期化してるってことかな）
      optimizerA.zero_grad()
      optimizerC.zero_grad()

      env.render()
      # state(4つの値を持つ配列)の生の値をfloat型のテンソルに変換している
      state = torch.FloatTensor(state).to(device)
      # ここで入力データが渡されているので、トレーニングが実行されて、forward関数が実行される
      # actorからはdistribution(状態に対する行動の確率分布)が出力される
      # criticからはvalue(行動価値関数？)が出力される
      dist, value = actor(state), critic(state)

      # actionは、Actorモデルから出力されたdistribution(ex: [0.5, 0.5])の中からサンプリングする
      # action: tensor(0, device='cuda:0') or action=tensor(1, device='cuda:0')
      # actionのテンソルはgpuに格納されている
      # ここでランダムに行動をサンプリングしているのは、モデルの学習時のデータを偏らせないようにするため
      # 最大確率の行動を選択していると、偏る
      action = dist.sample()
      # action.cpu(): tensor(0)
      # action.cpu().numpy(): 1
      # cpu().numpy()によって、テンソルはnumpyに変換されて、cpuに格納される
      next_state, reward, done, _ = env.step(action.cpu().numpy())

      # torchでは、distributionsに対して、log_prob関数が用意されており、サンプリングされたアクションを引数とすることで
      # 損失関数に必要なlog(pi(state))が求められる
      log_prob = dist.log_prob(action).unsqueeze(0)
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
      masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
      cumulated_rewards[-1] += reward
      experience = pd.DataFrame({'state': ','.join(map(str, state.cpu().numpy())), 'action': action.cpu().numpy(), 
                                'reward': reward, 'next_state': ','.join(map(str, next_state))}, index=[0])
      experiences = pd.concat([experiences, experience], ignore_index=True)
      state = next_state

      if done:
        print('Iteration: {}, Score: {}'.format(iter, i))
        succeeds.append(True)
        break


    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)
    returns = compute_returns(next_value, rewards, masks)
    log_probs = torch.cat(log_probs)
    # print(returns)  # cat前は、「[tensor([58.7051], device='cuda:0', grad_fn=<AddBackward0>), tensor([59])], 」がたくさんある
    # catによって複数あったtensorが一つに統一される「tensor([58, 59], device='cuda:0', grad_fn=<AddBackward0>)」
    # detachによって、同一のテンソルを持つ新しいテンソルがgrad等はfalseで作成される
    returns = torch.cat(returns).detach()
    # print(returns)
    values = torch.cat(values)

    advantage = returns - values

    # actor_loss: tensor(-1.4582, device='cuda:0', grad_fn=<NegBackward0>) 
    # tensorでできており、値があるのと、勾配を計算するためのgrad_fnがある
    actor_loss = -(log_probs * advantage.detach()).mean()
    # ciritcの損失関数には、平均二乗誤差を用いている
    critic_loss = advantage.pow(2).mean()

    # Save the losses for plot
    actor_losses.append(actor_loss.detach().item())
    critic_losses.append(critic_loss.detach().item())

    # stepを呼び出す前にbackward()を呼び出して勾配を計算する必要がある
    actor_loss.backward()
    critic_loss.backward()
    # stepメソッドによって、パラメータを更新する
    optimizerA.step()
    optimizerC.step()
  torch.save(actor, 'model/actor.pkl')
  torch.save(critic, 'model/critic.pkl')
  save_results('train', cumulated_rewards, succeeds, experiences, actor_losses, critic_losses)

  env.close()


if __name__ == '__main__':
  # .pklファイルがどうやらモデルらしい。モデルがあったら、それをloadして、なかったらActorクラスから作成している
  if os.path.exists('model/actor.pkl'):
    actor = torch.load('model/actor.pkl')
    print('Actor Model loaded')
  else:
    actor = Actor(state_size, action_size).to(device)
  if os.path.exists('model/critic.pkl'):
    critic = torch.load('model/critic.pkl')
    print('Critic Model loaded')
  else:
    critic = Critic(state_size, action_size).to(device)
  print(actor)
  print(critic)
  trainIters(actor, critic, n_iters=100)