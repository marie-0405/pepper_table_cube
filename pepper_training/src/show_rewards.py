#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import rospkg

rewards = []

# Set the logging system
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('pepper_training')
readfile = pkg_path + '/training_results/rewards.txt'

with open(readfile, 'r') as f:
  for line in f:
    rewards.append(float(line.rstrip('\n')))
nepisodes = np.arange(1, len(rewards) + 1)

plt.figure()
plt.plot(nepisodes, rewards)
plt.show()

