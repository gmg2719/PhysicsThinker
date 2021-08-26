#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def street_demo():
    # The geometry map is downloaded from the https://www.openstreetmap.org/
    ru_map = plt.imread('street.png')
    # longitude and latitude box for the area
    bbox = [104.06133, 104.07095, 30.54470, 30.54946]
    # Track positions of the moving
    longitude = [104.06410, 104.06400, 104.06390, 104.06380, 104.06370, 104.06360, 104.06350, 104.06340]
    latitude = [30.54736, 30.54736, 30.54736, 30.54736, 30.54736, 30.54736, 30.54736, 30.54736]
    # Draw the background geometry map and track positions
    fig,ax = plt.subplots(figsize = (8, 7))
    ax.scatter(longitude, latitude, zorder=1, alpha=0.8, c='r', s=10)
    ax.set_title('Plotting spatial data on Street Map')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(ru_map, zorder=0, extent=bbox, aspect='equal')

if __name__ == "__main__":
    print("Unit test")
    street_demo()
