from itertools import count
import gym
import json
import nep
import os
import time

import torch
import torch.optim as optim

from actor import Actor
from critic import Critic

def get_msg():
  while True:
    s, msg = sub.listen()
    if s:
      print(msg)
      return msg
    else:
      time.sleep(.0001)

# Create a new nep node
node = nep.node("Calculator")                                                       
conf = node.hybrid("192.168.0.101")                         
sub = node.new_sub("env", "json", conf)
pub = node.new_pub("calc", "json", conf) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = get_msg().values()
lr = 0.0001  # 学習率

def compute_returns(next_value, rewards, masks, gamma=0.99):
  # TD法を用いて、報酬から期待値（価値）を算出する
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def trainIters(actor, critic, n_iters):
  # parameters()の中身はtensorがいくつかある
  optimizerA = optim.Adam(actor.parameters())
  optimizerC = optim.Adam(critic.parameters())
  for iter in range(n_iters):
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    state = get_msg()['state']

    for i in range(5):
      # stateの生の値をfloat型のテンソルに変換している
      state = torch.FloatTensor(state).to(device)
      # ここで入力データが渡されているので、トレーニングが実行されて、forward関数が実行される
      # actorからはdistribution(状態に対する行動の確率分布)が出力される
      # criticからはvalue(行動価値関数？)が出力される
      dist, value = actor(state), critic(state)

      # actionは、Actorモデルから出力されたdistribution(ex: [0.5, 0.5])の中からサンプリングする
      # action: tensor(0, device='cuda:0') or action=tensor(1, device='cuda:0')
      # actionのテンソルはgpuに格納されている
      action = dist.sample()
      # action.cpu(): tensor(0)
      # action.cpu().numpy(): 1
      # cpu().numpy()によって、テンソルはnumpyに変換されて、cpuに格納される
      pub.publish({'action': action.cpu().numpy().tolist()})  # numpyのままだと送信できないので、tolist()が必要
      msg = get_msg()
      next_state = msg['next_state']
      reward = msg['reward']
      done = msg['done']

      log_prob = dist.log_prob(action).unsqueeze(0)
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
      masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

      state = next_state

      if done:
        print('Iteration: {}, Score: {}'.format(iter, i))
        break


    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)
    # エピソードごとに期待値を算出する
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # 最適化されたすべての勾配を0にする（初期化してるってことかな）
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    # stepを呼び出す前にbackward()を呼び出して勾配を計算する必要がある
    actor_loss.backward()
    critic_loss.backward()
    # stepメソッドによって、パラメータを更新する
    optimizerA.step()
    optimizerC.step()
  torch.save(actor, 'model/actor.pkl')
  torch.save(critic, 'model/critic.pkl')


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
  trainIters(actor, critic, n_iters=2)
