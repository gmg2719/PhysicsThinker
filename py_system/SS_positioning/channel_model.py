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

