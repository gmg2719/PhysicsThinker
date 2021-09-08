#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import math
import numpy as np
import os
import re
import sys

import scipy.constants as spy_constants
import scipy.signal as signal
from tdoa import Sim3DCord, find_most_average, kalman_filter, kalman_filter_protype
from tdoa import tdoa_positioning_4bs_improve, tdoa_positioning_5bs_assist
from channel_model import signal2distance, path_loss


def position_calc(bs1, bs2, bs3, bs4, bs5, sig1, sig2, sig3, sig4, sig5):
    light_speed = spy_constants.speed_of_light
    d1 = signal2distance(sig1, 120.0, 1)
    d2 = signal2distance(sig2, 120.0, 1)
    d3 = signal2distance(sig3, 120.0, 1)
    d4 = signal2distance(sig4, 120.0, 1)
    d5 = signal2distance(sig5, 120.0, 1)
    dt21 = (d2 - d1) / light_speed
    dt31 = (d3 - d1) / light_speed
    dt41 = (d4 - d1) / light_speed
    dt51 = (d5 - d1) / light_speed
    position = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51, 
                                           0, 0, 0, method='taylor-direct')
    return position

def faked_app_singleUe_outdoors(method):
    light_speed = spy_constants.speed_of_light
    tx_power = 120.0
    print('Outdoors simulation !')
    bs1 = Sim3DCord(1000, 1900, 60)
    bs2 = Sim3DCord(1000, 1000, 80)
    bs3 = Sim3DCord(1900, 1000, 10)
    bs4 = Sim3DCord(500, 1000, 40)
    bs5 = Sim3DCord(700, 800, 50)
    # There exists only one UE
    ue_x = 2000.0 * np.random.random()
    ue_y = 2000.0 * np.random.random()
    ue_z = 2 + 198.0 * np.random.random()
    ue = Sim3DCord(ue_x, ue_y, ue_z)
    ratio1 = 0
    ratio2 = 0
    ratio10 = 0
    dist1 = bs1.calc_distance(ue)
    dist2 = bs2.calc_distance(ue)
    dist3 = bs3.calc_distance(ue)
    dist4 = bs4.calc_distance(ue)
    dist5 = bs5.calc_distance(ue)

    iters = 1000
    samples = 20

    signal1 = tx_power - path_loss(dist1, 2.6, 1)
    signal2 = tx_power - path_loss(dist2, 2.6, 1)
    signal3 = tx_power - path_loss(dist3, 2.6, 1)
    signal4 = tx_power - path_loss(dist4, 2.6, 1)
    signal5 = tx_power - path_loss(dist5, 2.6, 1)
    np.random.seed(1)
    errors1 = np.random.randn(iters * samples)
    np.random.seed(3)
    errors2 = np.random.randn(iters * samples)
    np.random.seed(5)
    errors3 = np.random.randn(iters * samples)
    np.random.seed(7)
    errors4 = np.random.randn(iters * samples)
    np.random.seed(9)
    errors5 = np.random.randn(iters * samples)
    
    for i in range(iters):
        x_est = 0.
        y_est = 0.
        z_est = 0.
        x_list = []
        y_list = []
        z_list = []
        
        dt21_kalman = []
        dt31_kalman = []
        dt41_kalman = []
        dt51_kalman = []
        
        for k in range(samples):
            print("******iter= %d, SRS_ind=%d**********" % (i, k))
            recv_sig1 = signal1 + errors1[i * samples + k]
            recv_sig2 = signal2 + errors2[i * samples + k]
            recv_sig3 = signal3 + errors3[i * samples + k]
            recv_sig4 = signal4 + errors4[i * samples + k]
            recv_sig5 = signal5 + errors5[i * samples + k]
            d1 = signal2distance(recv_sig1, tx_power, 1)
            d2 = signal2distance(recv_sig2, tx_power, 1)
            d3 = signal2distance(recv_sig3, tx_power, 1)
            d4 = signal2distance(recv_sig4, tx_power, 1)
            d5 = signal2distance(recv_sig5, tx_power, 1)
            t1 = d1/light_speed 
            t2 = d2/light_speed
            t3 = d3/light_speed
            t4 = d4/light_speed
            t5 = d5/light_speed
            dt21 = t2 - t1
            dt41 = t4 - t1
            dt31 = t3 - t1
            dt51 = t5 - t1
            dt21_kalman.append(dt21)
            dt31_kalman.append(dt31)
            dt41_kalman.append(dt41)
            dt51_kalman.append(dt51)
            position = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51, 
                                                   0, 0, 0, method='taylor-direct')
            if (position.x < 0) or (position.y < 0) or (position.z < 0) or (
                np.isnan(position.x)) or (np.isnan(position.y) or (np.isnan(position.z))):
                continue
            x_list.append(position.x)
            y_list.append(position.y)
            z_list.append(position.z)

        if method.lower() == 'kalman':
            dt21_filter = kalman_filter_protype(samples, dt21_kalman, np.mean(dt21_kalman))[samples-1]
            dt31_filter = kalman_filter_protype(samples, dt31_kalman, np.mean(dt31_kalman))[samples-1]
            dt41_filter = kalman_filter_protype(samples, dt41_kalman, np.mean(dt41_kalman))[samples-1]
            dt51_filter = kalman_filter_protype(samples, dt51_kalman, np.mean(dt51_kalman))[samples-1]
            pos = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21_filter, dt31_filter, dt41_filter, dt51_filter, 
                                              0, 0, 0, method='taylor-direct')
            x_est = pos.x
            y_est = pos.y
            z_est = pos.z
        elif method.lower() == 'mean':
            dt21_filter = np.mean(dt21_kalman)
            dt31_filter = np.mean(dt31_kalman)
            dt41_filter = np.mean(dt41_kalman)
            dt51_filter = np.mean(dt51_kalman)
            pos = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21_filter, dt31_filter, dt41_filter, dt51_filter, 
                                              0, 0, 0, method='taylor-direct')
            x_est = pos.x
            y_est = pos.y
            z_est = pos.z
        elif method.lower() == 'wiener':
            dt21_filter = np.mean(signal.wiener(dt21_kalman, noise=1.0))
            dt31_filter = np.mean(signal.wiener(dt31_kalman, noise=1.0))
            dt41_filter = np.mean(signal.wiener(dt41_kalman, noise=1.0))
            dt51_filter = np.mean(signal.wiener(dt51_kalman, noise=1.0))
            pos = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21_filter, dt31_filter, dt41_filter, dt51_filter, 
                                              0, 0, 0, method='taylor-direct')
            x_est = pos.x
            y_est = pos.y
            z_est = pos.z
        elif method.lower() == 'likely':
            x_est = find_most_average(x_list)
            y_est = find_most_average(y_list)
            z_est = find_most_average(z_list)
        else:
            x_est = find_most_average(x_list)
            y_est = find_most_average(y_list)
            z_est = find_most_average(z_list)

        print("====The results====")
        print('UE estimate coordinate is (%.6f %.6f %.6f)' % (x_est, y_est, z_est))
        if (abs(x_est - ue_x) < 1.0) and (abs(y_est - ue_y) < 1.0) and (abs(z_est - ue_z) < 1.0):
            ratio1 += 1
        if (abs(x_est - ue_x) < 2.0) and (abs(y_est - ue_y) < 2.0) and (abs(z_est - ue_z) < 2.0):
            ratio2 += 1  
        if (abs(x_est - ue_x) < 10.0) and (abs(y_est - ue_y) < 10.0) and (abs(z_est - ue_z) < 10.0):
            ratio10 += 1
    print('UE real coordinate is (%.6f %.6f %.6f)' % (ue_x, ue_y, ue_z))
    print('ratio (CDF < 1.0m)  := %.4f' % (ratio1/iters))
    print('ratio (CDF < 2.0m)  := %.4f' % (ratio2/iters))
    print('ratio (CDF < 10.0m) := %.4f' % (ratio10/iters))

def faked_app_run_outdoors_sampling(tx_power, dist, steps=1000):
    signal_est = []
    signal_basic = tx_power - (28.0 + 22 * np.log10(dist) + 20 * np.log10(2.6))
    errors = np.random.randn(steps) * 0.707
    for i in range(steps):
        se = signal_basic + errors[i]
        signal_est.append(se)
    np.savetxt('signal_outdoor_estimation.txt', signal_est)
    return signal_est

