#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import numpy as np

import scipy.fftpack
import scipy as spy
from basic_sequences import LOWPAPRs

UL_REFERENCE_SEQUENCES_LEN = np.array([6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 
                                       90, 96, 108, 120, 132, 144, 150, 156, 162, 168,
                                       180, 192, 204, 216, 228, 240, 252, 264, 270, 276, 288,
                                       300, 312, 324, 336, 348, 360, 384, 396, 408, 432,
                                       450, 456, 480, 486, 504, 528, 540, 552, 576, 600, 624,
                                       648, 672, 696, 720, 750, 768, 792, 810, 816, 864,
                                       900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152, 1200, 1248,
                                       1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584, 1620, 1632])

SRS_BANDWIDTH_CONFIG_TABLE = np.array([[0, 	4,	    1,	4,	    1,	   4,	   1,	4,	1],
                                    [1, 	8,	    1,	4,	    2,	   4,	   1,	4,	1],
                                    [2, 	12,	    1,	4,	    3,	   4,	   1,	4,	1],
                                    [3, 	16,	    1,	4,	    4,	   4,	   1,	4,	1],
                                    [4, 	16,	    1,	8,	    2,	   4,	   2,	4,	1],
                                    [5, 	20,	    1,	4,	    5,	   4,	   1,	4,	1],
                                    [6, 	24,	    1,	4,	    6,	   4,	   1,	4,	1],
                                    [7, 	24,	    1,	12,	    2,	   4,	   3,	4,	1],
                                    [8, 	28,	    1,	4,	    7,	   4,	   1,	4,	1],
                                    [9, 	32,	    1,	16,	    2,	   8,	   2,	4,	2],
                                    [10,	36,	    1,	12,	    3,	   4,	   3,	4,	1],
                                    [11,	40,	    1,	20,	    2,	   4,	   5,	4,	1],
                                    [12,	48,	    1,	16,	    3,	   8,	   2,	4,	2],
                                    [13,	48,	    1,	24,	    2,	   12,	   2,	4,	3],
                                    [14,	52,	    1,	4,	    13,	   4,	   1,	4,	1],
                                    [15,	56,	    1,	28,	    2,	   4,	   7,	4,	1],
                                    [16,	60,	    1,	20,	    3,	   4,	   5,	4,	1],
                                    [17,	64,	    1,	32,	    2,	   16,	   2,	4,	4],
                                    [18,	72,	    1,	24,	    3,	   12,	   2,	4,	3],
                                    [19,	72,	    1,	36,	    2,	   12,	   3,	4,	3],
                                    [20,	76,	    1,	4,	    19,	   4,	   1,	4,	1],
                                    [21,	80,	    1,	40, 	2,	   20,	   2,	4,	5],
                                    [22,	88,	    1,	44, 	2,	   4,	   11,	4,	1],
                                    [23,	96,	    1,	32, 	3,	   16,	   2,	4,	4],
                                    [24,	96,	    1,	48, 	2,	   24,	   2,	4,	6],
                                    [25,	104,	1,	52, 	2,	   4,	   13,	4,	1],
                                    [26,	112,	1,	56, 	2,	   28,	   2,	4,	7],
                                    [27,	120,	1,	60, 	2,	   20,	   3,	4,	5],
                                    [28,	120,	1,	40, 	3,	   8,	   5,	4,	2],
                                    [29,	120,	1,	24, 	5,	   12,	   2,	4,	3],
                                    [30,	128,	1,	64, 	2,	   32,	   2,	4,	8],
                                    [31,	128,	1,	64, 	2,	   16,	   4,	4,	4],
                                    [32,	128,	1,	16, 	8,	   8,	   2,	4,	2],
                                    [33,	132,	1,	44, 	3,	   4,	   11,	4,	1],
                                    [34,	136,	1,	68, 	2,	   4,	   17,	4,	1],
                                    [35,	144,	1,	72, 	2,	   36,	   2,	4,	9],
                                    [36,	144,	1,	48, 	3,	   24,	   2,	12,	2],
                                    [37,	144,	1,	48, 	3,	   16,	   3,	4,	4],
                                    [38,	144,	1,	16, 	9,	   8,	   2,	4,	2],
                                    [39,	152,	1,	76, 	2,	   4,	   19,	4,	1],
                                    [40,	160,	1,	80, 	2,	   40,	   2,	4,	10],
                                    [41,	160,	1,	80, 	2,	   20,	   4,	4,	5],
                                    [42,	160,	1,	32, 	5,	   16,	   2,	4,	4],
                                    [43,	168,	1,	84, 	2,	   28,	   3,	4,	7],
                                    [44,	176,	1,	88, 	2,	   44,	   2,	4,	11],
                                    [45,	184,	1,	92, 	2,	   4,	   23,	4,	1],
                                    [46,	192,	1,	96, 	2,	   48,	   2,	4,	12],
                                    [47,	192,	1,	96, 	2,	   24,	   4,	4,	6],
                                    [48,	192,	1,	64, 	3,	   16,	   4,	4,	4],
                                    [49,	192,	1,	24, 	8,	   8,	   3,	4,	2],
                                    [50,	208,	1,	104,	2,	   52,	   2,	4,	13],
                                    [51,	216,	1,	108,	2,	   36,	   3,	4,	9],
                                    [52,	224,	1,	112,	2,	   56,	   2,	4,	14],
                                    [53,	240,	1,	120,	2,	   60,	   2,	4,	15],
                                    [54,	240,	1,	80, 	3,	   20,	   4,	4,	5],
                                    [55,	240,	1,	48, 	5,	   16,	   3,	8,	2],
                                    [56,	240,	1,	24, 	10,	   12,	   2,	4,	3],
                                    [57,	256,	1,	128,	2,	   64,	   2,	4,	16],
                                    [58,	256,	1,	128,	2,	   32,	   4,	4,	8],
                                    [59,	256,	1,	16, 	16,	   8,	   2,	4,	2],
                                    [60,	264,	1,	132, 	2,	   44,	   3,	4,	11],
                                    [61,	272,	1,	136, 	2,	   68,	   2,	4,	17],
                                    [62,	272,	1,	68, 	4,	   4,	   17,	4,	1],
                                    [63,	272,	1,	16, 	17,	   8,	   2,	4,	2]])

def srs_freq_start_pos(frame_number, slot_number, srs_config, ttis_per_subframe):
    k_tc_p = 0
    k_0_overbar_p = 0
    N_b = 0
    m_srs_b = 0
    n_srs = 0
    f_b = 0
    n_b = 0;
    k_tc_overbar = srs_config.comb_offset
    k_tc = srs_config.comb_size
    c_srs = srs_config.config_index
    b_srs = srs_config.bandwidth_index
    b_hop = srs_config.frequency_hopping
    # /* it adjusts the SRS allocation to align with the common resource block grid in multiples of four */
    n_rrc = srs_config.frequency_position
    n_shift = srs_config.frequency_shift
    # Now, only repetitions = 0 is supported
    num_repetitions = 0
    R = 2**num_repetitions
    t_offset = srs_config.t_offset
    t_srs = srs_config.t_srs
    # /* consecutive OFDM symbols */
    n_symb_srs = srs_config.num_symbols
    l = 0;
    
    k_tc_p = k_tc_overbar
    # Now, only BWP start = 0 is supported
    k_0_p = n_shift * 12 + k_tc_p + 0

    for b in range(b_srs+1):
        n_b = 0;
        N_b = SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2*b + 2]
        m_srs_b = SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2*b + 1]

        if (b_hop >= b_srs):
            n_b = int(4 * n_rrc / m_srs_b) % N_b
        else:
            n_b = (4 * n_rrc / m_srs_b) % N_b

            if (b > b_hop):
                if (srs_config.resource_type == 0):
                    n_srs = l / R;
                else:
                    n_srs = ((20 * frame_number + slot_number - t_offset) / t_srs) * (n_symb_srs / R) + (l / R)

                product_n_b = 1
                for b_prime in np.arange(b_hop+1, b_srs):
                    product_n_b *= SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2 * b_prime + 2]

                if (N_b % 2 == 1):
                    f_b = int(N_b / 2) * int(n_srs / product_n_b)
                else:
                    product_n_b_b_srs = product_n_b
                    product_n_b_b_srs *= SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2 * b_srs + 2]
                    f_b = (N_b / 2) * ((n_srs % product_n_b_b_srs) / product_n_b) + ((n_srs % product_n_b_b_srs) / 2 * product_n_b)
                n_b = int(f_b + (4 * n_rrc / m_srs_b)) % N_b
        k_0_p += m_srs_b * 12 * n_b                                     

    return k_0_p

def srs_scramble_data(data_in, length, c_init, data_out):
    x = np.zeros(2, dtype=np.uint64)
    x[0] = np.uint64(5188146772878295041)
    x1 = x[0]

    x[1] = c_init
    x2 = x[1]
    x2 = (x2 | ((x2 ^ (x2 >> np.uint64(1)) ^ (x2 >> np.uint64(2)) ^ (x2 >> np.uint64(3))) << np.uint64(31))) & np.uint64(0x7FFFFFFFFFFFFFF)
    x2 = x2 ^ (((x2 >> np.uint64(28)) ^ (x2 >> np.uint64(29)) ^ (x2 >> np.uint64(30)) ^ (x2 >> np.uint64(31))) << np.uint64(59))

    for i in np.arange(1, 25):
        x1 = (x1 >> np.uint64(2)) ^ (x1 >> np.uint64(8))
        x1 = x1 ^ (x1 << np.uint64(56)) ^ (x1 << np.uint64(62))
        x2 = (x2 >> np.uint64(2)) ^ (x2 >> np.uint64(4)) ^ (x2 >> np.uint64(6)) ^ (x2 >> np.uint64(8))
        x2 = x2 ^ (x2 << np.uint64(56)) ^ (x2 << np.uint64(58)) ^ (x2 << np.uint64(60)) ^ (x2 << np.uint64(62))

    num = ((np.uint32(length) - np.uint32(1)) >> np.uint32(6)) + np.uint32(1)
    out_tmp = data_out.view(dtype = np.uint64)
    in_tmp = data_in.view(dtype = np.uint64)
    for i in range(num):
        x1 = (x1 >> np.uint64(2)) ^ (x1 >> np.uint64(8))
        x1 = x1 ^ (x1 << np.uint64(56)) ^ (x1 << np.uint64(62))
        x2 = (x2 >> np.uint64(2)) ^ (x2 >> np.uint64(4)) ^ (x2 >> np.uint64(6)) ^ (x2 >> np.uint64(8))
        x2 = x2 ^ (x2 << np.uint64(56)) ^ (x2 << np.uint64(58)) ^ (x2 << np.uint64(60)) ^ (x2 << np.uint64(62))
        out_tmp[i] = in_tmp[i] ^ x1 ^ x2
        # print('TMP %lu %lu' %(out_tmp[i], in_tmp[i]))
    out = out_tmp.view(dtype=np.uint8)
    for i in range(length):
        data_out[i] = out[i]

def srs_group_sequence_hopping(srs_config, symb_idx, slot_idx, m_sc_b_srs):
    u = 0
    v = 0
    # generate PN sequence
    index = 0
    f_gh = 0
    num_bits = 2240;
    
    group_or_sequence_hopping = srs_config.group_or_sequence_hopping
    hopping_id = srs_config.sequence_id
    
    bit_in = np.zeros(183456, dtype=np.uint8)
    random_seq = np.zeros(num_bits, dtype=np.uint8)
    srs_scramble_data(bit_in, num_bits, hopping_id, random_seq)
    
    if (group_or_sequence_hopping.lower() == 'group_hopping'):
        f_gh = random_seq[slot_idx * 14 + symb_idx] % 30
    elif (group_or_sequence_hopping.lower() == 'sequence_hopping'):
        if (m_sc_b_srs >= 6 * 12):
            index = (slot_idx * 14 + symb_idx)
            v = ((random_seq[index >> 3] & (1 << (index & 7))) >> (index & 7))
    u = int(f_gh + hopping_id) % 30
    
    return u, v

def srs_get_cyclic_shift(point_offset, point_len, cyclic_shift_value, total_len, vec_in, vec_out):
    for i in range(total_len):
        vec_out[i] = 0
    for ii in range(point_len):
        if point_offset + ii < cyclic_shift_value :
            t1 = (point_offset + ii + total_len - cyclic_shift_value)
        else:
            t1 = (point_offset + ii - cyclic_shift_value)
        if point_offset + ii < total_len :
            t2 = (point_offset + ii)
        else:
            t2 = (point_offset + ii - total_len)
        if t2 < 0:
            t2 = t2 + total_len
        vec_out[t1] = vec_in[t2]

def srs_ch_estimates_proc(srs_config, grid, srs_pilot, nb_re_srs, n_ap, cycliShift, k_tc, nb_antennas_rx, nfft):
    aarx = 0
    n_srs_cs_max = 0
    
    # TS 38.211 6.4.1.4.2 Sequence generation
    if k_tc == 4:
        n_srs_cs_max = 12
    elif k_tc == 2:
        n_srs_cs_max = 8
    else:
        return
    
    min_cs_diff = n_srs_cs_max
    n_srs_cs_i = cycliShift
    for p_index1 in range(n_ap):
        for p_index2 in range(n_ap):
            cs_diff = n_srs_cs_i[p_index1] - n_srs_cs_i[p_index2]
            if (p_index1 != p_index2) and (abs(cs_diff) < min_cs_diff):
                min_cs_diff = abs(cs_diff)
    
    len_per_cs = int(nb_re_srs/n_srs_cs_max)
    win_size = min(4, min_cs_diff) * len_per_cs
    rms = 32
    peak_pos = 0
    peak_size = int((rms * nb_re_srs * k_tc - 1) / nfft) + 2
    peak_offset = 0
    if (nb_re_srs * k_tc * 32 < nfft):
        peak_size = max(int(win_size/2), 3)
    
    # Frequency domain data
    srs_ch_estimates_freq = np.zeros((nb_antennas_rx, nb_re_srs), dtype=complex)
    # Time domain data
    srs_ch_estimates_time = np.zeros((nb_antennas_rx, nb_re_srs), dtype=complex)
    srs_vars_tmp = np.zeros(nb_re_srs, dtype=float)
    srs_ch_estimates_time_pow = np.zeros(nb_re_srs, dtype=float)
    srs_ch_estimates_time_pow3 = np.zeros(3 * nb_re_srs, dtype=float)
    for aarx in range(nb_antennas_rx):
        # Freq channel estimation, z = x * y.conj()
        for k in range(nb_re_srs):
            srs_ch_estimates_freq[aarx][k] = grid[aarx][k] * np.conj(srs_pilot[k])
        # IDFT to get time channel estimation
        srs_ch_estimates_time[aarx] = nb_re_srs * spy.fft.ifft(srs_ch_estimates_freq[aarx])

        # Get the time channel estimation Power and RSSI
        for k in range(nb_re_srs):
            srs_vars_tmp[k] = np.real(srs_ch_estimates_time[aarx][k]) * np.real(
                srs_ch_estimates_time[aarx][k]) + np.imag(srs_ch_estimates_time[aarx][k]) * np.imag(srs_ch_estimates_time[aarx][k])
            srs_ch_estimates_time_pow[k] += srs_vars_tmp[k]
    srs_rssi = np.sum(srs_ch_estimates_time_pow) / nb_re_srs
    
    # 3 points sum for peak detection
    if (nb_re_srs * k_tc * 32) < nfft:
        for k in range(nb_re_srs):
            srs_ch_estimates_time_pow3[nb_re_srs + k] = srs_ch_estimates_time_pow[k]
    else:
        # z[i] = r[i] + (x[i] + y[i])
        for k in range(nb_re_srs - 2):
            srs_ch_estimates_time_pow3[nb_re_srs + k + 1] = srs_ch_estimates_time_pow[k + 2] + (
                srs_ch_estimates_time_pow[k] + srs_ch_estimates_time_pow[k + 1])
        srs_ch_estimates_time_pow3[nb_re_srs] = srs_ch_estimates_time_pow[nb_re_srs - 1] + srs_ch_estimates_time_pow[0] + srs_ch_estimates_time_pow[1]
        srs_ch_estimates_time_pow3[2 * nb_re_srs - 1] = srs_ch_estimates_time_pow[nb_re_srs - 2] + srs_ch_estimates_time_pow[nb_re_srs - 1] + srs_ch_estimates_time_pow[0]
    for k in range(nb_re_srs):
        srs_ch_estimates_time_pow3[k] = srs_ch_estimates_time_pow3[nb_re_srs + k]
        srs_ch_estimates_time_pow3[2*nb_re_srs+k] = srs_ch_estimates_time_pow3[nb_re_srs + k]
    
    rssi_sum = srs_rssi
    # Cyclicshift process : TA estimation and power calculation
    peak_sc_sum = 0
    srs_hpow = np.zeros(n_ap, dtype=float)
    srs_TOest = np.zeros(n_ap, dtype=float)
    srs_TOsum = np.zeros(n_ap, dtype=complex)
    srs_ch_estimates_freq_pow = np.zeros((n_ap, nb_re_srs), dtype=float)
    srs_ch_estimates = np.zeros((n_ap*nb_antennas_rx, nb_re_srs), dtype=complex)
    srs_vars_tmp = np.zeros(nb_re_srs, dtype=complex)
    srs_vars_tmp1 = np.zeros(nb_re_srs, dtype=float)
    for p_index in range(n_ap):
        offset_per_cs = int(len_per_cs * ((n_srs_cs_max - n_srs_cs_i[p_index]) % n_srs_cs_max))
        
        # Get peak position
        max_pos = 0
        max_value = srs_ch_estimates_time_pow3[nb_re_srs+ offset_per_cs - int(win_size / 2)]
        for k in range(win_size+1):
            if (srs_ch_estimates_time_pow3[nb_re_srs+ offset_per_cs - int(win_size / 2)+k] > max_value):
                max_pos = k
                max_value = srs_ch_estimates_time_pow3[nb_re_srs+ offset_per_cs - int(win_size / 2)+k]
        peak_pos = max_pos + offset_per_cs - int(win_size / 2)
        peak_offset = peak_pos - (int((peak_size - 1) / 4) + 1)
        srs_TOest[p_index] = peak_pos - offset_per_cs
        
        # Remove cyclicshift
        for aarx in range(nb_antennas_rx):
            srs_get_cyclic_shift(peak_offset, peak_size, offset_per_cs, nb_re_srs, srs_ch_estimates_time[aarx], srs_vars_tmp)
            
            # get freq channel estimation
            srs_ch_estimates[p_index * nb_antennas_rx + aarx] = spy.fft.fft(srs_vars_tmp)
            
            # get freq channel estimation power
            for k in range(nb_re_srs):
                e = srs_ch_estimates[p_index * nb_antennas_rx + aarx][k]
                srs_vars_tmp1[k] = np.real(e) * np.real(e) + np.imag(e) * np.imag(e)
            # Z = X + Y * h
            for k in range(nb_re_srs):
                h = 1.0/(nb_re_srs * nb_re_srs)
                srs_ch_estimates_freq_pow[p_index][k] = srs_ch_estimates_freq_pow[p_index][k] + srs_vars_tmp1[k] * h
            
            if nb_re_srs * k_tc * 16 < nfft:
                for k in range(nb_re_srs - 1):
                    srs_TOsum[p_index] += srs_ch_estimates[p_index * nb_antennas_rx + aarx][k] * np.conj(srs_ch_estimates[p_index * nb_antennas_rx + aarx][k + 1])

        srs_hpow[p_index] = 0.
        for k in range(nb_re_srs):
            srs_hpow[p_index] += srs_ch_estimates_freq_pow[p_index][k]
        rssi_sum -= srs_hpow[p_index]
        peak_sc_sum += peak_size
        # print(srs_TOsum[p_index], srs_hpow[p_index], rssi_sum, peak_sc_sum)
    
    # Get the noise power
    srs_noisepow = rssi_sum / (nb_re_srs - peak_sc_sum) / nb_antennas_rx
    
    re_per_rb = int(12 / k_tc)
    nb_rb = int(nb_re_srs / re_per_rb)
    srs_wideband_snr = np.zeros(n_ap, dtype=float)
    # Get the SINR of each RB
    for p_index in range(n_ap):
        if (nb_re_srs * k_tc * 16) < nfft:
            srs_TOest[p_index] = np.arctan2(np.imag(srs_TOsum[p_index]), np.real(srs_TOsum[p_index])) / (2 * np.pi * nb_re_srs * k_tc)
        else:
            srs_TOest[p_index] /= (nb_re_srs * k_tc)

        srs_wideband_snr[p_index] = 0
        tmp_sum = 0.
        for rb_idx in range(nb_rb):
            tmp_sum = 0.
            for k in range(re_per_rb):
                tmp_sum += srs_ch_estimates_freq_pow[p_index][rb_idx * re_per_rb + k]
            srs_wideband_snr[p_index] += (tmp_sum / re_per_rb / srs_noisepow)
        srs_wideband_snr[p_index] /= nb_rb
    # Collect the results
    print(srs_TOest)
    print(srs_wideband_snr)
    # The estimation of wideband SNR and TA
    sig_est = 0.
    ta_est = 0.
    for srs_cs_idx in range(srs_config.num_ports):
        sig_est += (1/srs_config.num_ports) * srs_wideband_snr[srs_cs_idx]
        ta_est += (1/srs_config.num_ports) * srs_TOest[srs_cs_idx]
    sig_est = 10.0 * math.log10(sig_est)    # dB
    return ta_est, sig_est

def srs_demodulation(sfn_id, slot_id, sys_parameters, srs_pdu, rx_grid):
    c_srs = srs_pdu.config_index
    b_srs = srs_pdu.bandwidth_index
    k_tc = srs_pdu.comb_size
    m_srs_b = SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2 * b_srs + 1]
    m_sc_b_srs = int(m_srs_b * 12 / k_tc)
    
    k_0_p = srs_freq_start_pos(sfn_id, slot_id, srs_pdu, 2)
    
    l_start = srs_pdu.start_symbol_index
    # There are totally 14 time symbols
    num_of_subcarriers = int(np.shape(rx_grid)[0] / 14)
    start_subcarrier = l_start * num_of_subcarriers + k_0_p

    m_sc_b_srs_index = 0
    while ((UL_REFERENCE_SEQUENCES_LEN[m_sc_b_srs_index] != m_sc_b_srs) and (m_sc_b_srs_index < 84)):
        m_sc_b_srs_index += 1

    u, v = srs_group_sequence_hopping(srs_pdu, l_start, slot_id, m_sc_b_srs)
    
    srs_pilot = np.zeros(m_sc_b_srs, dtype=complex)

    # low_p1 = LOWPAPRs(0, 0, [], 6).gen_base()
    # low_p2 = LOWPAPRs(0, 0, [], 12).gen_base()
    # low_p3 = LOWPAPRs(0, 0, [], 18).gen_base()
    low_p = LOWPAPRs(u, v, [], m_sc_b_srs).gen_base()
    for i in range(m_sc_b_srs):
        srs_pilot[i] = low_p[i]

    nb_antennas_rx = sys_parameters.rx_ants
    rx_grid_ext = np.zeros((nb_antennas_rx, m_sc_b_srs), dtype=complex)
    for aarx in range(nb_antennas_rx):
        subcarrier = start_subcarrier
        for k in range(m_sc_b_srs):
            rx_grid_ext[aarx][k] = rx_grid[subcarrier][aarx]
            subcarrier += k_tc
    
    nfft = sys_parameters.nfft
    nrof_cyclicShift = srs_pdu.num_ports
    cycliShift = np.zeros(nrof_cyclicShift, dtype=int)
    SRS_antenna_port = [1000, 1001, 1002, 1003]
    if k_tc == 4:
        n_srs_cs_max = 12
    elif k_tc == 2:
        n_srs_cs_max = 8
    
    for p_index in range(nrof_cyclicShift):
        cycliShift[p_index] = (srs_pdu.cyclic_shift + (n_srs_cs_max * int(SRS_antenna_port[p_index] - 1000) / srs_pdu.num_ports)) % n_srs_cs_max
    
    ta_est, sig_est = srs_ch_estimates_proc(srs_pdu, rx_grid_ext, srs_pilot, m_sc_b_srs, 
                                            nrof_cyclicShift, cycliShift, k_tc, nb_antennas_rx, nfft)
    return ta_est, sig_est

def srs_rx_proc(sfn_id, slot_id, sys_parameters, srs_pdu, rx_grid):
    if np.size(srs_pdu) != 1:
        print('Only support 1 SRS PDU right now !')
        sys.exit(-1)
    ta_est, sig_est = srs_demodulation(sfn_id, slot_id, sys_parameters, srs_pdu, rx_grid)
    print('Estimate SRS (TA, Sigal_strength) : %.8f %.8f' % (ta_est, sig_est))

