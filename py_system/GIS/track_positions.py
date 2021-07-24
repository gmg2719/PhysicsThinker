#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib.animation import FuncAnimation

_RADIO_SCALE = 100

def draw_BSs(x, y):
    x_min = min(x) - _RADIO_SCALE
    x_max = max(x) + _RADIO_SCALE
    y_min = min(y) - _RADIO_SCALE
    y_max = max(y) + _RADIO_SCALE
    low_pos = min(x_min, y_min)
    high_pos = max(x_max, y_max)
    fig, ax = plt.subplots()
    image = plt.imread('./bs.jpg')
    im = OffsetImage(image, zoom=0.2)
    ax.scatter(x, y, zorder=10, s=0)
    plt.xlim(low_pos, high_pos)
    plt.ylim(low_pos, high_pos)
    for x0, y0, in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
    return ax

def draw_UEs(ax, x, y):
    print('UE number = ', len(x))
    ax.scatter(x, y, marker='o', s=0.5, c='r', zorder=20)

def draw_track_UEs(x, y, velocity=1.0):
    # Generate the random UEs distribution
    xx = [random.uniform(0, 500) for i in range(100)]
    yy = [random.uniform(0, 500) for i in range(100)]
    # Every 2 seconds to view each UEs, total 100 seconds
    cost_time = 0
    while cost_time <= 10.0:
        print("dt = ", cost_time)
        xx, yy = track_randomWalk_UEs(2.0, xx, yy, velocity)
        ax = draw_BSs(x, y)
        draw_UEs(ax, xx, yy)
        cost_time += 2.0

def track_randomWalk_UEs(dt, x, y, velocity=1.0):
    x0 = np.array(x.copy())
    y0 = np.array(y.copy())
    # 4 directions random walk model
    for i in range(len(x0)):
        # when val = 9, 1/9, the UE keep the position at this dt period 
        val = random.randint(1, 9)
        if (val == 1) or (val == 2):
            x0[i] = x0[i] + dt * velocity
        elif (val == 3) or (val == 4):
            x0[i] = x0[i] - dt * velocity
        elif (val == 5) or (val == 6):
            y0[i] = y0[i] + dt * velocity
        elif (val == 7) or (val == 8):
            y0[i] = y0[i] - dt * velocity
    return x0, y0

if __name__ == "__main__":
    print("Unit test")
    x = [50, 250, 450]
    y = [50, 250, 50]
    draw_track_UEs(x, y, 4)

