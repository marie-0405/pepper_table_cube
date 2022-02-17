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

    return hand, cube, target

if __name__ == '__main__':
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)

    # hand, = ax.plot([], [], ms=50, label='手先')
    # cube, = ax.plot([], [], ms=50, label='キューブ')
    # target, = ax.plot([], [], ms=50, label='目標')

    hand, = ax.plot([], [], label='手先')
    cube, = ax.plot([], [], label='キューブ')
    target, = ax.plot([], [], label='目標')

    ani = FuncAnimation(fig, animate, frames=np.arange(0, len(time)), interval=50, blit=True)

    ax.set_xlim(-1, 1.25)
    ax.set_ylim(-1, 1.0)

    # Set the label
    ax.set_xlabel("x座標[m]", size = 14)
    ax.set_ylabel("y座標[m]", size = 14)

    ax.grid()
    plt.legend()
    plt.show()

    ani.save('./test/data/data.gif', writer='pillow', fps=50)