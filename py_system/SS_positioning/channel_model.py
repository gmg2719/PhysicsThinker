#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np

def signal2distance(sig, tx_power, mode):
    if mode == 0:
        return 10**((tx_power - sig - 32.4 - 20 * np.log10(2.6)) / 17.3)
    elif mode == 1:
        return 10**((tx_power - sig - 28 - 20 * np.log10(2.6)) / 22)
    else:
        return 0.

def path_loss(dist, freq, mode):
    loss = 0
    if mode == 0:
        # 2.6 GHz center frequency
        loss = 32.4 + 17.3 * np.log10(dist) + 20 * np.log10(freq)
    elif mode == 1:
        loss = 28.0 + 22 * np.log10(dist) + 20 * np.log10(freq)
    else:
        print("ts_link_srs_positioning() only support indoor or outdoor layout !")
    return loss

def find_most_average(e_list):
    size = np.size(e_list)
    if size == 0:
        return 0.
    near_counter = np.zeros(np.size(e_list), dtype=int)
    for i in range(size):
        e = e_list[i]
        for k in range(size):
            if abs(e-e_list[k]) < 5.0:
                near_counter[i] += 1
    indices = near_counter.argmax()
    basic_value = e_list[indices]
    average = 0.
    counter = 0
    for i in range(size):
        if abs(e_list[i] - basic_value) < 5.0:
            average += e_list[i]
            counter += 1
    return average / counter
