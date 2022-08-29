import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


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
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    env.reset()

    for i in count():
      env.render()
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
      next_state, reward, done, _ = env.step(action.cpu().numpy())

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
  print([i for i in actor.parameters()])
  print(critic)
  trainIters(actor, critic, n_iters=100)