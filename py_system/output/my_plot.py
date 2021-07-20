#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def my_plot_polyline(x, y):
    fig, ax = plt.subplots()
    x = np.array(x)
    y = np.array(y)
    ax.plot(x, y)

def my_plot_points(x, y):
    fig, ax = plt.subplots()
    x = np.array(x)
    y = np.array(y)
    ax.scatter(x, y)

if __name__ == "__main__":
    print("Unit test")
    my_plot_polyline([1, 2, 3, 4], [1, 4, 2, 3])
    my_plot_points([1, 2, 3], [4, 5, 6])
