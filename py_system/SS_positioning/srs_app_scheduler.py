#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import math
import numpy as np
import os
import re
import sys

from basic_parameters import SysParameters
from ofdm_wave import WaveForm, ofdm_modulate, ofdm_demodulate
from srs_receiver import srs_rx_proc
from srs_sender import srs_tx_proc

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

    # Combine the received OFDM wave in the time domain
    rx_wave = np.array(np.zeros((np.shape(rx_wave_tmp)[0], np.shape(rx_wave_tmp)[1]), dtype=complex))
    for rx_idx in range(1):
        for tx_idx in range(1):
            rx_wave[:,rx_idx] += rx_wave_tmp[:, rx_idx, tx_idx]

    # Pass the channel as there exists the path-loss
    loss = 0
    if mode == 0:
        # 2.6 GHz center frequency
        loss = 32.4 + 17.3 * np.log10(dist) + 20 * np.log10(freq)
    elif mode == 1:
        loss = 28.0 + 22 * np.log10(dist) + 20 * np.log10(freq)
    else:
        print("ts_link_srs_positioning() only support indoor or outdoor layout !")
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
    srs_rx_proc(0, 7, sys_parameters, srs_pdu, rx_grid)
    print('SRS run done')

