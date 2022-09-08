from itertools import count
import json
import nep
import os
import time

import torch
import torch.optim as optim

from actor import Actor
from critic import Critic
from result_controller import ResultController
import settings

def get_msg():
  while True:
    s, msg = sub.listen()
    if s:
      print(msg)
      return msg

# Create a new nep node
node = nep.node("Calculator")                                                       
conf = node.hybrid("192.168.3.14")                         
sub = node.new_sub("env", "json", conf)
pub = node.new_pub("calc", "json", conf) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size, state_size = get_msg().values()

def compute_returns(next_value, rewards, masks):
  # compute returns with rewards and next value bu Temporal Differential method
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + settings.gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def trainIters(actor, critic):
  optimizerA = optim.Adam(actor.parameters())
  optimizerC = optim.Adam(critic.parameters())

  # Initialize the result data
  cumulated_rewards = []
  succeeds = []
  actor_losses = []
  critic_losses = []

  for iter in range(settings.nepisodes):
    cumulated_rewards.append(0)
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    state = get_msg()['state']

    for i in range(settings.nsteps):
      state = torch.FloatTensor(state).to(device)
      dist, value = actor(state), critic(state)
      action = dist.sample()

      pub.publish({'action': action.cpu().numpy().tolist()})  # need tolist for sending message as json
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
      cumulated_rewards[-1] += reward

      if done:
        print('Iteration: {}, Score: {}'.format(iter, i))
        break
      
      # Judgement for End or Not
      if done:
        succeeds.append(True)
        break
      else:
        if i == settings.nsteps - 1:
          succeeds.append(False)
        else:
          state = next_state

    next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)

    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # Save the losses for plot
    actor_losses.append(actor_loss.detach().item())
    critic_losses.append(critic_loss.detach().item())

    # Optimize the weight parameters
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()  # calculate gradient
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

  torch.save(actor, 'model/actor.pkl')
  torch.save(critic, 'model/critic.pkl')

  ## Save the results
  result_controller = ResultController("actor-critic")
  result_controller.write(cumulated_rewards, succeeds=succeeds, actor_losses=actor_losses, critic_losses=critic_losses)
  result_controller.plot('reward')
  result_controller.plot('actor_loss')
  result_controller.plot('critic_loss')


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
  trainIters(actor, critic)
