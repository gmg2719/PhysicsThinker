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
from track_positions import track_randomWalk_UEs

_RADIO_SCALE = 100
x = [50, 250, 450]
y = [50, 250, 50]
xx = []
yy = []

class AnimatedUEsTrack(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.numpoints = numpoints

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=2000, frames=100,
                                           init_func=self.setup_plot)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        global x, y
        global xx, yy
        xx = [random.uniform(0, 500) for i in range(self.numpoints)]
        yy = [random.uniform(0, 500) for i in range(self.numpoints)]
        self.scat = self.ax.scatter(x, y, zorder=10, s=0)
        self.ax.axis([-150, 650, -150, 650])
        # Set the BSs icon
        image = plt.imread('./bs.jpg')
        im = OffsetImage(image, zoom=0.2)
        for x0, y0, in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), frameon=False)
            self.ax.add_artist(ab)
        self.scat = self.ax.scatter(xx, yy, zorder=20, s=2.0, c='r')
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def update(self, i):
        global xx, yy
        """Update the scatter plot."""
        xx, yy = track_randomWalk_UEs(2.0, xx, yy, 5.0)
        data = np.stack((xx, yy), axis=1)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # self.scat = self.ax.scatter(xx, yy, zorder=20, s=1.0, c='r')
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
    
    def get_ani(self):
        return self.ani

def animation_track_UEs():
    a = AnimatedUEsTrack(1)
    plt.show()
    #ani = a.get_ani()
    #ani.save('track.gif', writer='imagemagick', fps=2)

if __name__ == "__main__":
    print("Unit test")
    animation_track_UEs()

