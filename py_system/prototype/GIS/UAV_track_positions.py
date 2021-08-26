#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation

class UAVsTrack(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, uavs=50):
        self.numpoints = uavs
        self.left = 0
        self.right = 30
        self.top = 30
        self.low = 0
        self.dt = 1.0           # delta_t = 1.0s for each discrete moment 
        self.velocity = 2.0     # 2 m/s for the UAV
        self.frames = 0

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        self.xx = [random.uniform(self.left, self.right) for i in range(self.numpoints)]
        self.yy = [random.uniform(self.low, self.top) for i in range(self.numpoints)]
        low = min(self.left, self.low)
        high = max(self.right, self.top)
        self.ax.axis([low, high, low, high])
        # Draw the background if it is necessary
        pass
        # Draw the UAVs
        image = plt.imread('./uav.jpg')
        im = OffsetImage(image, zoom=0.05)
        for x0, y0, in zip(self.xx, self.yy):
            ab = AnnotationBbox(im, (x0, y0), frameon=False)
            self.ax.add_artist(ab)
        self.scat = self.ax.scatter(self.xx, self.yy, zorder=20, s=0)
        # Then setup FuncAnimation. (3 minutes, interval = 1000 ms)
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=20, interval=500)

    def update(self, i):
        """Update the scatter plot."""
        if self.frames > 20:
            self.ani.event_source.stop()
        self.frames += 1
        
        dt = self.dt;
        vec = self.velocity
        low = min(self.left, self.low)
        high = max(self.right, self.top)
        # 4 directions random walk model
        for i in range(len(self.xx)):
            val = random.randint(1, 4)
            if val == 1:
                self.xx[i] = self.xx[i] + dt * vec
            elif val == 2:
                self.xx[i] = self.xx[i] - dt * vec
            elif val == 3:
                self.yy[i] = self.yy[i] + dt * vec
            elif val == 4:
                self.yy[i] = self.yy[i] - dt * vec
            # Check if the UAV run out of the boundary
            if (self.xx[i] <= low):
                self.xx[i] = low + 1.0
            if (self.xx[i] >= high):
                self.xx[i] = high - 1.0
            if (self.yy[i] <= low):
                self.yy[i] = low + 1.0
            if (self.yy[i] >= high):
                self.yy[i] = high - 1.0

        data = np.stack((self.xx, self.yy), axis=1)

        # Set x and y data...
        for ann in self.ax.artists:
            ann.remove()
        self.ax.artists = []
        print('Frame : ', self.frames)
        # Draw the UAVs
        image = plt.imread('./uav.jpg')
        im = OffsetImage(image, zoom=0.05)
        for i in range(len(self.xx)):
            x0 = self.xx[i]
            y0 = self.yy[i]
            ab = AnnotationBbox(im, (x0, y0), frameon=False)
            self.ax.add_artist(ab)
            print('add one UAV !')
        # self.scat.set_offsets(data[:, :2])
        plt.pause(0.25)
        self.fig.canvas.draw()
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.ax.artists,

    def get_ani(self):
        return self.ani       


if __name__ == "__main__":
    print("Unit test")
    a = UAVsTrack(2)
    plt.show()
    # ani = a.get_ani()
    # ani.save('uav.gif', writer='imagemagick', fps=4)

