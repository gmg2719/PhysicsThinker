#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import re
import sys

import scipy.constants as spy_constants
import scipy.signal as signal
from tdoa import Sim3DCord, kalman_filter_protype, tdoa_positioning_4bs_improve, tdoa_positioning_5bs_assist
from channel_model import signal2distance

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

def position_calc2(bs1, bs2, bs3, bs4, sig1, sig2, sig3, sig4):
    light_speed = spy_constants.speed_of_light
    d1 = signal2distance(sig1, 80.0, 0)
    d2 = signal2distance(sig2, 80.0, 0)
    d3 = signal2distance(sig3, 80.0, 0)
    d4 = signal2distance(sig4, 80.0, 0)
    dt21 = (d2 - d1) / light_speed
    dt31 = (d3 - d1) / light_speed
    dt41 = (d4 - d1) / light_speed
    position = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, dt21, dt31, dt41,
                                            0, 0, 0, method='taylor-direct')
    return position

def kalman_filter_protype(n_iter, z, init_guess):
    sz = (n_iter, )

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 10**1.0 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = init_guess
    P[0] = 1000
    
    for k in range(1,n_iter):
        # measurement update
        K[k] = P[k-1]/( P[k-1]+R )
        xhat[k] = xhat[k-1] + K[k] * (z[k-1] - xhat[k-1])
        P[k] = (1-K[k])*P[k-1]
    return xhat

def kalman_example():
    # Kalman positioning example
    position = 50.0
    x_measure = [48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95]
    x_estimate = kalman_filter_protype(10, x_measure, 60.0)

def outdoors_example():    
    y1 = 11.382345292050346
    y1_measure = [13.006690655713587, 10.77058887840027, 10.85417353978689, 10.309376669894174, \
                 12.247752921375024, 9.080806595170063, 13.127157056266826, 10.621138391155243, \
                 11.701384388107444, 11.132974916572936]
    # np.random.seed(1)
    # y1_measure -= np.random.randn(10)
    y1_estimate =  kalman_filter_protype(10, y1_measure, np.mean(y1_measure))[9]
    
    y2 = 16.768515297912927
    y2_measure = [18.557143771343245, 17.205025148424916, 16.865012765984936, 14.905022594548436, \
                  16.49112709539853, 16.41375631864394, 16.685773816430466, 16.14151462108908, \
                  16.724697128937, 16.291297267553425]
    # np.random.seed(3)
    # y2_measure -= np.random.randn(10)
    y2_estimate = kalman_filter_protype(10, y2_measure, np.mean(y2_measure))[9]
    
    y3 = 17.421188702988147
    y3_measure = [17.862416189873187, 17.090318551094057, 19.851959889995925, 17.16909657338507, \
                  17.53079854456633, 19.00366982004971, 16.511956298131906, 16.82955204505786, \
                  17.608791928825184, 17.09131874520879]
    # np.random.seed(5)
    # y3_measure -= np.random.randn(10)
    y3_estimate = kalman_filter_protype(10, y3_measure, np.mean(y3_measure))[9]
    
    y4 = 14.357147417904486
    y4_measure = [16.047673121704843, 13.891210047363654, 14.38996758158307, 14.764663700900993, \
                  13.568224389278747, 14.359212990810434, 14.356257032046555, 12.602423111559064, \
                  15.37480542356798, 14.957645933824034]
    # np.random.seed(7)
    # y4_measure -= np.random.randn(10)
    y4_estimate = kalman_filter_protype(10, y4_measure, np.mean(y4_measure))[9]
    
    y5 = 16.47900838168546
    y5_measure = [16.48011693639767, 16.189464312435263, 15.362942078621046, 16.466125624939668, \
                  16.100646917477544, 15.997873018808557, 14.961677202889788, 15.988136400463715, \
                  16.238327803182788, 15.831060921463857]
    # np.random.seed(9)
    # y5_measure -= np.random.randn(10)
    y5_estimate = kalman_filter_protype(10, y5_measure, np.mean(y5_measure))[9]
    
    bs1 = Sim3DCord(1000, 1900, 60)
    bs2 = Sim3DCord(1000, 1000, 80)
    bs3 = Sim3DCord(1900, 1000, 10)
    bs4 = Sim3DCord(500, 1000, 40)
    bs5 = Sim3DCord(700, 800, 50)
    
    y1_estimate = np.mean(signal.wiener(y1_measure, noise=1.0))
    y2_estimate = np.mean(signal.wiener(y2_measure, noise=1.0))
    y3_estimate = np.mean(signal.wiener(y3_measure, noise=1.0))
    y4_estimate = np.mean(signal.wiener(y4_measure, noise=1.0))
    y5_estimate = np.mean(signal.wiener(y5_measure, noise=1.0))
    pos = position_calc(bs1, bs2, bs3, bs4, bs5, y1_estimate, y2_estimate, y3_estimate, y4_estimate, y5_estimate)
    print('Filter Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))
    pos = position_calc(bs1, bs2, bs3, bs4, bs5, np.mean(y1_measure), np.mean(y2_measure), np.mean(y3_measure),
                        np.mean(y4_measure), np.mean(y5_measure))
    print('Average Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))
    pos = position_calc(bs1, bs2, bs3, bs4, bs5, y1, y2, y3, y4, y5)
    print('Real Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))
    print("==========================================")
    for i in range(20):
        y1 += np.random.rand()
        y2 += np.random.rand()
        y3 += np.random.rand()
        y4 += np.random.rand()
        y5 += np.random.rand()
        pos = position_calc(bs1, bs2, bs3, bs4, bs5, y1, y2, y3, y4, y5)

def position_calc_time(bs1, bs2, bs3, bs4, bs5, toa1, toa2, toa3, toa4, toa5):
    dt21 = toa2 - toa1
    dt31 = toa3 - toa1
    dt41 = toa4 - toa1
    dt51 = toa5 - toa1
    position = tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51, 
                                           0, 0, 0, method='taylor-direct')
    return position

def outdoors_example2():
    bs1 = Sim3DCord(1000, 1900, 60)
    bs2 = Sim3DCord(1000, 1000, 80)
    bs3 = Sim3DCord(1900, 1000, 10)
    bs4 = Sim3DCord(500, 1000, 40)
    bs5 = Sim3DCord(700, 800, 50)
    minor = 8.138E-10
    toa1 = 6.462040108726696e-06
    toa2 = 3.6774242865123566e-06
    toa3 = 3.4346047559719065e-06
    toa4 = 4.733162018399112e-06
    toa5 = 3.79055799092721e-06
    
    print("Reference position is :")
    pos = position_calc_time(bs1, bs2, bs3, bs4, bs5, toa1, toa2, toa3, toa4, toa5)
    print("==========================================")
    toa1_list = []
    toa2_list = []
    toa3_list = []
    toa4_list = []
    toa5_list = []
    samples = 20
    for i in range(samples):
        toa1 += np.random.rand() * 8.133E-9
        toa2 += np.random.rand() * 8.133E-9
        toa3 += np.random.rand() * 8.133E-9
        toa4 += np.random.rand() * 8.133E-9
        toa5 += np.random.rand() * 8.133E-9
        toa1_list.append(toa1)
        toa2_list.append(toa2)
        toa3_list.append(toa3)
        toa4_list.append(toa4)
        toa5_list.append(toa5)
    toa1 = kalman_filter_protype(samples, toa1_list, np.mean(toa1_list))[samples-1]
    toa2 = kalman_filter_protype(samples, toa2_list, np.mean(toa2_list))[samples-1]
    toa3 = kalman_filter_protype(samples, toa3_list, np.mean(toa3_list))[samples-1]
    toa4 = kalman_filter_protype(samples, toa4_list, np.mean(toa4_list))[samples-1]
    toa5 = kalman_filter_protype(samples, toa5_list, np.mean(toa5_list))[samples-1]
    pos = position_calc_time(bs1, bs2, bs3, bs4, bs5, toa1, toa2, toa3, toa4, toa5)
        # print('Real Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))

def indoors_example():
    y1 = 16.523734436378234
    y1_measure = [16.488987963403716, 16.627404287162115, 16.38166920613236, 16.702280640859456, 16.230449172611564, 16.402331457903085, 16.60887610126999, 16.399217595544066, 16.520163932202216, 16.549064791296512]
    y1_estimate =  kalman_filter_protype(10, y1_measure, np.mean(y1_measure))[9]
    
    y2 = 18.748120587934153
    y2_measure = [18.71299540973438, 18.84716654056011, 18.601686832658693, 18.92173609981699, 18.45943923703919, 18.620717600733336, 18.832401317546623, 18.62267586005809, 18.73649938202763, 18.774259897051728]
    y2_estimate = kalman_filter_protype(10, y2_measure, np.mean(y2_measure))[9]
    
    y3 = 27.251989127351536
    y3_measure = [27.2173557991743, 27.342039195588992, 27.09716040782287, 27.416229356582946, 26.97420759926784, 27.112380928241958, 27.335667256112934, 27.12552909261057, 27.22448155140459, 27.281227316578736]
    y3_estimate = kalman_filter_protype(10, y3_measure, np.mean(y3_measure))[9]
    
    y4 = 18.013140761783134
    y4_measure = [17.97808924411319, 18.113557546888362, 17.867998862032042, 18.188211248546203, 17.723033430227417, 17.887533273374025, 18.097643031085106, 17.8879462968106, 18.00391341016958, 18.038992304104056]
    y4_estimate = kalman_filter_protype(10, y4_measure, np.mean(y4_measure))[9]

    bs1 = Sim3DCord(10, 19, 6)
    bs2 = Sim3DCord(1, 1, 8)
    bs3 = Sim3DCord(19, 1, 10)
    bs4 = Sim3DCord(5, 10, 4)
    pos = position_calc2(bs1, bs2, bs3, bs4, y1_estimate, y2_estimate, y3_estimate, y4_estimate)
    print('Kalman Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))
    pos = position_calc2(bs1, bs2, bs3, bs4, np.mean(y1_measure), np.mean(y2_measure), np.mean(y3_measure),
                        np.mean(y4_measure))
    print('Average Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))
    pos = position_calc2(bs1, bs2, bs3, bs4, y1, y2, y3, y4)
    print('Real Position solution is (%.6f %.6f %.6f)' % (pos.x, pos.y, pos.z))

outdoors_example2()
