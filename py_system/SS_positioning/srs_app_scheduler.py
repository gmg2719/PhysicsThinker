#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import math
import numpy as np
import os
import re
import sys

import scipy.constants as spy_constants
from basic_parameters import SysParameters
from ofdm_wave import WaveForm, ofdm_modulate, ofdm_demodulate
from srs_receiver import srs_rx_proc, srs_rx_proc2
from srs_sender import srs_tx_proc
from tdoa import Sim3DCord, find_most_average, kalman_filter
from tdoa import tdoa_positioning_4bs_improve, tdoa_positioning_5bs_assist
from channel_model import signal2distance, path_loss

def srs_app_positioning_schedule(sys_parameters, srs_pdu, txpower, dist, mode, freq=2.6):
    """
    Run link simulation of the uplink SRS procedure.
        mode = 0, indoors; mode = 1, outdoors
        freq = 2.6 GHz
        txpower = send power of the UE
        dist = the propagation distance between the UE and the BaseStation
    """
    # Construct the TX OFDM signal for the uplink system
    # Perform TX operations
    tx_grid = np.zeros((sys_parameters.n_re_total, sys_parameters.n_symb_total, 
                        sys_parameters.tx_ants), dtype=complex)

    tx_grid += srs_tx_proc(0, 7, sys_parameters, srs_pdu)

    #
    #  1. OFDM modulate
    #
    # The OFDM signal go through the simplified path loss channel model
    waveform = WaveForm()
    tx_wave = ofdm_modulate(sys_parameters, waveform, tx_grid)
    print("OFDM modulate : generate the signal done !")
    
    #
    #  2. PASS the channel
    #
    # According to the TX and RX antenna, calculate the OFDM wave from the view of the receiver
    ant_gain =  [[1,1],[1,-1]]
    rx_wave_tmp = np.array(np.zeros((np.shape(tx_wave)[0], sys_parameters.rx_ants, sys_parameters.tx_ants), dtype=complex))
    for tx_idx in range(sys_parameters.tx_ants):
        for rx_idx in range(sys_parameters.rx_ants):
            rx_wave_tmp[:, rx_idx, tx_idx] = ant_gain[rx_idx][tx_idx] * tx_wave[:, tx_idx]

    # Add time delay, SamplingRate = 122880000, SamplingPoints = 4096
    offset = int(dist / spy_constants.speed_of_light * 122.88e6)
    if offset >= 1:
        rx_wave_tmp = np.roll(rx_wave_tmp, offset)
        rx_wave_tmp[0:offset,:,:] = 0

    # Combine the received OFDM wave in the time domain
    rx_wave = np.array(np.zeros((np.shape(rx_wave_tmp)[0], np.shape(rx_wave_tmp)[1]), dtype=complex))
    for rx_idx in range(1):
        for tx_idx in range(1):
            rx_wave[:,rx_idx] += rx_wave_tmp[:, rx_idx, tx_idx]

    # Pass the channel as there exists the path-loss
    loss = path_loss(dist, freq, mode)

    # apply awgn noise
    total_size = np.shape(rx_wave)[0] * np.shape(rx_wave)[1]
    noise = np.random.randn(total_size) + np.random.randn(total_size) * 1j
    noise = noise / math.sqrt(2.0)
    if (np.shape(rx_wave)[1] == 1):
        noise = noise.reshape((-1, 1))
    snr = 10**((txpower - loss)/20)
    rx_wave = snr * rx_wave + noise
    print("SRS reference value : %.6f" % (txpower - loss))

    #
    #  3. OFDM demodulate
    #
    rx_grid = ofdm_demodulate(sys_parameters, waveform, rx_wave)
    rx_grid = np.reshape(rx_grid, (sys_parameters.n_re_total * sys_parameters.n_symb_total, sys_parameters.rx_ants), order='F').copy()
    print("OFDM demodulate : demodulate the signal done !")
    
    # Perform RX operations
    ta_est, signal_est = srs_rx_proc(0, 7, sys_parameters, srs_pdu, rx_grid)
    print('SRS run done')
    
    return ta_est, signal_est

def srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist, steps=1000):
    signal_est = []
    ta_est = []
    for i in range(steps):
        te, se = srs_app_positioning_schedule(sys_parameters, srs_pdu, tx_power, dist, 0)
        signal_est.append(se)
        ta_est.append(te)
    np.savetxt('signal_indoor_estimation.txt', signal_est)
    np.savetxt('ta_indoor_estimation.txt', ta_est)
    return ta_est, signal_est

def srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist, steps=1000):
    signal_est = []
    ta_est = []
    for i in range(steps):
        te, se = srs_app_positioning_schedule(sys_parameters, srs_pdu, tx_power, dist, 1)
        signal_est.append(se)
        ta_est.append(te)
    np.savetxt('signal_outdoor_estimation.txt', signal_est)
    np.savetxt('ta_outdoor_estimation.txt', ta_est)
    return ta_est, signal_est



#
#
#
#  The simulation of the single UE
#
#
#
def srs_app_singleUe_indoors(sys_parameters, srs_pdu, tx_power):
    light_speed = spy_constants.speed_of_light
    print('Indoors simulation !')
    bs1 = Sim3DCord(10, 19, 6)
    bs2 = Sim3DCord(1, 1, 8)
    bs3 = Sim3DCord(19, 1, 10)
    bs4 = Sim3DCord(5, 10, 4)
    # There exists only one UE
    ue_x = 20.0 * np.random.random()
    ue_y = 20.0 * np.random.random()
    ue_z = 2 + 18.0 * np.random.random()
    ue = Sim3DCord(ue_x, ue_y, ue_z)
    
    dist1 = bs1.calc_distance(ue)
    dist2 = bs2.calc_distance(ue)
    dist3 = bs3.calc_distance(ue)
    dist4 = bs4.calc_distance(ue)

    ratio1 = 0
    ratio2 = 0
    ratio10 = 0
    iters = 1
    samples = 10

    np.random.seed(1)
    ta1, signal1 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist1, iters * samples)
    np.random.seed(2)
    ta2, signal2 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist2, iters * samples)
    np.random.seed(3)
    ta3, signal3 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist3, iters * samples)
    np.random.seed(4)
    ta4, signal4 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist4, iters * samples)

    for i in range(iters):
        x_est = 0.
        y_est = 0.
        z_est = 0.
        x_list = []
        y_list = []
        z_list = []
        
        for k in range(samples):
            print("******iter= %d, SRS_ind=%d**********" % (i, k))
            signal_est1 = signal1[samples * i + k]
            signal_est2 = signal2[samples * i + k]
            signal_est3 = signal3[samples * i + k]
            signal_est4 = signal4[samples * i + k]
            
            d1 = signal2distance(signal_est1, tx_power, 0)
            d2 = signal2distance(signal_est2, tx_power, 0)
            d3 = signal2distance(signal_est3, tx_power, 0)
            d4 = signal2distance(signal_est4, tx_power, 0)
            
            dt21 = (d2 - d1) / light_speed
            dt31 = (d3 - d1) / light_speed
            dt41 = (d4 - d1) / light_speed
            position = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, dt21, dt31, dt41, 0, 0, 0, method='taylor-direct')
            if (position.x < 0) or (position.y < 0) or (position.z < 0) or (
                np.isnan(position.x)) or (np.isnan(position.y) or (np.isnan(position.z))):
                continue
            print("***%d*** : %.6f %.6f %.6f" %(k, position.x, position.y, position.z))
            x_list.append(position.x)
            y_list.append(position.y)
            z_list.append(position.z)
        x_est = find_most_average(x_list)
        y_est = find_most_average(y_list)
        z_est = find_most_average(z_list)
        # print('UE REF (%.6f %.6f %6f), Estimate (%.6f %.6f %.6f)' %(ue_x, ue_y, ue_z, x_est, y_est, z_est))
        if (abs(x_est - ue_x) < 1.0) and (abs(y_est - ue_y) < 1.0) and (abs(z_est - ue_z) < 1.0):
            ratio1 += 1
        if (abs(x_est - ue_x) < 2.0) and (abs(y_est - ue_y) < 2.0) and (abs(z_est - ue_z) < 2.0):
            ratio2 += 1  
        if (abs(x_est - ue_x) < 10.0) and (abs(y_est - ue_y) < 10.0) and (abs(z_est - ue_z) < 10.0):
            ratio10 += 1
    print('UE real coordinate is (%.6f %.6f %.6f)' % (ue_x, ue_y, ue_z))
    print('ratio := %.4f %.4f %.4f' % (ratio1/iters, ratio2/iters, ratio10/iters))

def srs_app_singleUe_outdoors(sys_parameters, srs_pdu, tx_power, method='power'):
    light_speed = spy_constants.speed_of_light
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

    iters = 1
    samples = 10
    
    np.random.seed(1)
    ta1, signal1 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist1, iters*samples)
    np.random.seed(2)
    ta2, signal2 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist2, iters*samples)
    np.random.seed(3)
    ta3, signal3 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist3, iters*samples)
    np.random.seed(4)
    ta4, signal4 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist4, iters*samples)
    np.random.seed(5)
    ta5, signal5 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist5, iters*samples)
    
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
            if method.lower() == 'power':
                signal_est1 = signal1[samples * i + k]
                signal_est2 = signal2[samples * i + k]
                signal_est3 = signal3[samples * i + k]
                signal_est4 = signal4[samples * i + k]
                signal_est5 = signal5[samples * i + k]
                d1 = signal2distance(signal_est1, tx_power, 1)
                d2 = signal2distance(signal_est2, tx_power, 1)
                d3 = signal2distance(signal_est3, tx_power, 1)
                d4 = signal2distance(signal_est4, tx_power, 1)
                d5 = signal2distance(signal_est5, tx_power, 1)
                dt21 = ta2[samples * i + k] - ta1[samples * i + k]
                dt31 = ta3[samples * i + k] - ta1[samples * i + k]
                dt41 = ta4[samples * i + k] - ta1[samples * i + k]
                dt51 = ta5[samples * i + k] - ta1[samples * i + k]
            else:
                # TA estimation mode
                t1 = ta1[samples * i + k]
                t2 = ta2[samples * i + k]
                t3 = ta3[samples * i + k]
                t4 = ta4[samples * i + k]
                t5 = ta5[samples * i + k]
                dt21 = t2 - t1
                dt31 = t3 - t1
                dt41 = t4 - t1
                dt51 = t5 - t1
                # print('Estimate : ', t1, t2, t3, t4, t5)
                # print('Real     : ', dist1/light_speed, dist2/light_speed, dist3/light_speed, dist4/light_speed)
   
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

        x_est = find_most_average(x_list)
        y_est = find_most_average(y_list)
        z_est = find_most_average(z_list)

        dt21_filter = kalman_filter(samples, dt21_kalman)
        dt31_filter = kalman_filter(samples, dt31_kalman)
        dt41_filter = kalman_filter(samples, dt41_kalman)
        dt51_filter = kalman_filter(samples, dt51_kalman)
        position = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21_filter[samples-1], dt31_filter[samples-1],
                                               dt41_filter[samples-1], dt51_filter[samples-1], 
                                               0, 0, 0, method='taylor-direct')
        print('UE estimate coordinate is (%.6f %.6f %.6f)' % (x_est, y_est, z_est))
        print("UE SRS Kalman Filter coordinate is (%.6f %.6f %.6f)" % (position.x, position.y, position.z))

        if (abs(x_est - ue_x) < 1.0) and (abs(y_est - ue_y) < 1.0) and (abs(z_est - ue_z) < 1.0):
            ratio1 += 1
        if (abs(x_est - ue_x) < 2.0) and (abs(y_est - ue_y) < 2.0) and (abs(z_est - ue_z) < 2.0):
            ratio2 += 1  
        if (abs(x_est - ue_x) < 20.0) and (abs(y_est - ue_y) < 20.0) and (abs(z_est - ue_z) < 20.0):
            ratio10 += 1
    print('UE real coordinate is (%.6f %.6f %.6f)' % (ue_x, ue_y, ue_z))
    print('ratio := %.4f %.4f %.4f' % (ratio1/iters, ratio2/iters, ratio10/iters))


#
#
#
#  Something about multiple UEs
#
#
#
def srs_app_multipleUe_indoors(sys_parameters, srs_pdu, tx_power, ue_num):
    light_speed = spy_constants.speed_of_light
    print('Indoors simulation !')
    bs1 = Sim3DCord(10, 19, 6)
    bs2 = Sim3DCord(1, 1, 8)
    bs3 = Sim3DCord(19, 1, 10)
    bs4 = Sim3DCord(5, 10, 4)

    if ue_num <= 0:
        print("srs_app_multipleUe_indoors() must be applied to multiple UEs !")
        return

    ratio1 = 0
    ratio2 = 0
    ratio10 = 0
    for i in range(ue_num):
        ue_x = 20.0 * np.random.random()
        ue_y = 20.0 * np.random.random()
        ue_z = 2 + 18.0 * np.random.random()
        ue = Sim3DCord(ue_x, ue_y, ue_z)

        x_est = 0.
        y_est = 0.
        z_est = 0.
        x_list = []
        y_list = []
        z_list = []
        dist1 = bs1.calc_distance(ue)
        dist2 = bs2.calc_distance(ue)
        dist3 = bs3.calc_distance(ue)
        dist4 = bs4.calc_distance(ue)
        
        np.random.seed(10)
        ta1, signal1 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist1, 10)
        np.random.seed(10)
        ta2, signal2 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist2, 10)
        np.random.seed(10)
        ta3, signal3 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist3, 10)
        np.random.seed(10)
        ta4, signal4 = srs_app_run_indoors_sampling(sys_parameters, srs_pdu, tx_power, dist4, 10)

        for k in range(10):
            print("******UE= %d, SRS_ind=%d**********" % (i, k))
            signal_est1 = signal1[k]
            signal_est2 = signal2[k]
            signal_est3 = signal3[k]
            signal_est4 = signal4[k]
            d1 = signal2distance(signal_est1, tx_power, 0)
            d2 = signal2distance(signal_est2, tx_power, 0)
            d3 = signal2distance(signal_est3, tx_power, 0)
            d4 = signal2distance(signal_est4, tx_power, 0)
            dt21 = (d2 - d1) / light_speed
            dt31 = (d3 - d1) / light_speed
            dt41 = (d4 - d1) / light_speed
            position = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, dt21, dt31, dt41, 
                                                    0, 0, 0, method='taylor-direct')
            if (position.x < 0) or (position.y < 0) or (position.z < 0) or (
                np.isnan(position.x)) or (np.isnan(position.y) or (np.isnan(position.z))):
                continue
            print("***%d*** : %.6f %.6f %.6f" %(k, position.x, position.y, position.z))
            x_list.append(position.x)
            y_list.append(position.y)
            z_list.append(position.z)
        x_est = find_most_average(x_list)
        y_est = find_most_average(y_list)
        z_est = find_most_average(z_list)

        print('UE REF (%.6f %.6f %6f), Estimate (%.6f %.6f %.6f)' %(ue_x, ue_y, ue_z, x_est, y_est, z_est))
        if (abs(x_est - ue_x) < 1.0) and (abs(y_est - ue_y) < 1.0) and (abs(z_est - ue_z) < 1.0):
            ratio1 += 1
        if (abs(x_est - ue_x) < 2.0) and (abs(y_est - ue_y) < 2.0) and (abs(z_est - ue_z) < 2.0):
            ratio2 += 1  
        if (abs(x_est - ue_x) < 10.0) and (abs(y_est - ue_y) < 10.0) and (abs(z_est - ue_z) < 10.0):
            ratio10 += 1
    print('ratio := %.4f %.4f %.4f' % (ratio1/ue_num, ratio2/ue_num, ratio10/ue_num))

def srs_app_multipleUe_outdoors(sys_parameters, srs_pdu, tx_power, ue_num):
    light_speed = spy_constants.speed_of_light
    print('Outdoors simulation !')
    bs1 = Sim3DCord(1000, 1900, 60)
    bs2 = Sim3DCord(1000, 1000, 80)
    bs3 = Sim3DCord(1900, 1000, 10)
    bs4 = Sim3DCord(500, 1000, 40)
    bs5 = Sim3DCord(700, 800, 50)
    
    if ue_num <= 0:
        print("srs_app_multipleUe_outdoors() must be applied to multiple UEs !")
        return
    
    ratio1 = 0
    ratio2 = 0
    ratio10 = 0

    for i in range(ue_num):
        ue_x = 2000.0 * np.random.random()
        ue_y = 2000.0 * np.random.random()
        ue_z = 2 + 198.0 * np.random.random()
        ue = Sim3DCord(ue_x, ue_y, ue_z)
    
        x_est = 0.
        y_est = 0.
        z_est = 0.
        x_list = []
        y_list = []
        z_list = []
        dist1 = bs1.calc_distance(ue)
        dist2 = bs2.calc_distance(ue)
        dist3 = bs3.calc_distance(ue)
        dist4 = bs4.calc_distance(ue)
        dist5 = bs5.calc_distance(ue)
        
        np.random.seed(10)
        ta1, signal1 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist1, 10)
        np.random.seed(10)
        ta2, signal2 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist2, 10)
        np.random.seed(10)
        ta3, signal3 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist3, 10)
        np.random.seed(10)
        ta4, signal4 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist4, 10)
        np.random.seed(10)
        ta5, signal5 = srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, tx_power, dist5, 10)
        
        for k in range(10):
            print("******UE= %d, SRS_ind=%d**********" % (i, k))
            signal_est1 = signal1[k]
            signal_est2 = signal2[k]
            signal_est3 = signal3[k]
            signal_est4 = signal4[k]
            signal_est5 = signal5[k]
            d1 = signal2distance(signal_est1, tx_power, 0)
            d2 = signal2distance(signal_est2, tx_power, 0)
            d3 = signal2distance(signal_est3, tx_power, 0)
            d4 = signal2distance(signal_est4, tx_power, 0)
            d5 = signal2distance(signal_est5, tx_power, 0)
            dt21 = (d2 - d1) / light_speed
            dt31 = (d3 - d1) / light_speed
            dt41 = (d4 - d1) / light_speed
            dt51 = (d5 - d1) / light_speed
            position = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51,
                                                   0, 0, 0, method='taylor-direct')
            if (position.x < 0) or (position.y < 0) or (position.z < 0) or (
                np.isnan(position.x)) or (np.isnan(position.y) or (np.isnan(position.z))):
                continue
            print("***%d*** : %.6f %.6f %.6f" %(k, position.x, position.y, position.z))
            x_list.append(position.x)
            y_list.append(position.y)
            z_list.append(position.z)
        x_est = find_most_average(x_list)
        y_est = find_most_average(y_list)
        z_est = find_most_average(z_list)
    
        print('UE REF (%.6f %.6f %6f), Estimate (%.6f %.6f %.6f)' %(ue_x, ue_y, ue_z, x_est, y_est, z_est))
        if (abs(x_est - ue_x) < 1.0) and (abs(y_est - ue_y) < 1.0) and (abs(z_est - ue_z) < 1.0):
            ratio1 += 1
        if (abs(x_est - ue_x) < 2.0) and (abs(y_est - ue_y) < 2.0) and (abs(z_est - ue_z) < 2.0):
            ratio2 += 1  
        if (abs(x_est - ue_x) < 10.0) and (abs(y_est - ue_y) < 10.0) and (abs(z_est - ue_z) < 10.0):
            ratio10 += 1
    print('ratio := %.4f %.4f %.4f' % (ratio1/ue_num, ratio2/ue_num, ratio10/ue_num))

