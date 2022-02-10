#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np

json_open = open('my_hopper_training/training_results/data/openaigym.episode_batch.0.19725.stats.json', 'r')
json_load = json.load(json_open)

rewards = json_load["episode_rewards"]
nepisodes = np.arange(1, len(rewards) + 1)

plt.figure()
plt.plot(nepisodes, rewards)
plt.show()

time_min = (json_load["timestamps"][-1] - json_load["initial_reset_timestamp"]) / 60
print("It takes about {:.0f} minutes".format(time_min))


