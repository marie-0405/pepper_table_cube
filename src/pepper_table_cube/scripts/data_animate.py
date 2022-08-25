#!/usr/bin/env python
# coding:utf-8
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

FILE_NAME = 'data'
RATE = 25.0

dirname = os.path.dirname(__file__)
df = pd.read_csv('./test/data/{}.csv'.format(FILE_NAME))

plt.rcParams["font.size"] = 20

hand_x = np.array(df['hand_x'])
hand_y = np.array(df['hand_y'])
cube_x = np.array(df['cube_x'])
cube_y = np.array(df['cube_y'])
target_x = np.array(df['target_x'])
target_y = np.array(df['target_y'])

stride = float(1 / RATE)
time = np.linspace(0, stride * df.shape[0], df.shape[0])


def animate(i): 
    hand.set_data(hand_x[i], hand_y[i])
    cube.set_data(cube_x[i], cube_y[i])
    target.set_data(target_x[i], target_y[i])

    return hand, cube, target,

if __name__ == '__main__':
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)

    hand, = ax.plot([], [], 'o', ms=50)
    cube, = ax.plot([], [], 's', ms=50)
    target, = ax.plot([], [], '*', ms=50)

    ani = FuncAnimation(fig, animate, frames=np.arange(0, len(time)), interval=50, blit=True)

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.5, 0.0)

    ## for legend
    # ax.scatter([], [], label='手先', color='b', marker='o')
    # ax.scatter([], [], label='目標', color='r', marker='s')
    # ax.scatter([], [], label='キューブ', color='g', marker='*')

    # Set the label
    ax.set_xlabel("x座標[m]", fontsize = 25)
    ax.set_ylabel("y座標[m]", fontsize = 25)

    ax.grid()
    plt.legend(loc='lower right')
    plt.show()

    ani.save('./test/data/data.gif', writer='pillow', fps=50)