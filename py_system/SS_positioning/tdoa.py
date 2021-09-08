#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy.constants as spy_constants
from scipy.optimize import fsolve
from scipy.optimize import leastsq

class Sim3DCord(object):
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def calc_distance(self, dest):
        return math.sqrt((self.x - dest.x)**2 + (self.y - dest.y)**2 + (self.z - dest.z)**2)
 
    def debug_print(self):
        print('Coordinate is : %.6f, %.6f, %.6f' % (self.x, self.y, self.z))

def kalman_filter(n_iter, z):
    sz = (n_iter, )
    Q = 1.0
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

def kalman_filter_protype(n_iter, z, init_guess):
    sz = (n_iter, )

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 1E-2 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = init_guess
    P[0] = 1.0

    for k in range(1,n_iter):
        # measurement update
        K[k] = P[k-1]/( P[k-1]+R )
        xhat[k] = xhat[k-1] + K[k] * (z[k-1] - xhat[k-1])
        P[k] = (1-K[k])*P[k-1]
    return xhat

def find_most_average(e_list):
    size = np.size(e_list)
    if size == 0:
        return 0.
    near_counter = np.zeros(np.size(e_list), dtype=int)
    for i in range(size):
        e = e_list[i]
        for k in range(size):
            if abs(e-e_list[k]) < 10.0:
                near_counter[i] += 1
    indices = near_counter.argmax()
    basic_value = e_list[indices]
    average = 0.
    counter = 0
    for i in range(size):
        if abs(e_list[i] - basic_value) < 10.0:
            average += e_list[i]
            counter += 1
    return average / counter

# From : 3D TDOA Problem Solution with Four Receiving Nodes, J. D. Gonzalez, R. Alvarez, etc., 27 June, 2019.
# dt21 : UE to bs2 and bs1 TOA difference
# dt31 : UE to bs3 and bs1 TOA difference
# dt41 : UE to bs4 and bs1 TOA difference
def tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, dt21, dt31, dt41, x_init, y_init, z_init, method='newton'):
    position = Sim3DCord(0.0, 0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    def equations_3d(p):
        x, y, z = p
        r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
        r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
        r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
        r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
        return (r2 - r1 - light_speed*dt21, r3 - r1 - light_speed*dt31, r4 - r1 - light_speed*dt41)
    
    def taylor_solver(bs1, bs2, bs3, bs4, L, R, U):
        x = 0
        y = 0
        z = 0
        XL = bs2.x - bs1.x
        YL = bs2.y - bs1.y
        ZL = bs2.z - bs1.z
        XR = bs3.x - bs1.x
        YR = bs3.y - bs1.y
        ZR = bs3.z - bs1.z
        XU = bs4.x - bs1.x
        YU = bs4.y - bs1.y
        ZU = bs4.z - bs1.z
        E = L*L - XL*XL - YL*YL - ZL*ZL
        F = R*R - XR*XR - YR*YR - ZR*ZR
        G = U*U - XU*XU - YU*YU - ZU*ZU
        delta = -8 * (XL*YR*ZU+XU*YL*ZR+XR*YU*ZL-XL*YU*ZR-XR*YL*ZU-XU*YR*ZL)
        delta1 = 4*(YR*ZU-YU*ZR)
        delta2 = 4*(YL*ZU-YU*ZL)
        delta3 = 4*(YL*ZR-YR*ZL)
        # print("delta := ", delta, delta1, delta2, delta3)
        MX = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NX = (1/delta)*(E*delta1-F*delta2+G*delta3)
        # print("MX, NX := ", MX, NX)
        delta1 = 4*(XR*ZU-XU*ZR)
        delta2 = 4*(XL*ZU-XU*ZL)
        delta3 = 4*(XL*ZR-XR*ZL)
        MY = (2/delta)*(-L*delta1+R*delta2-U*delta3)
        NY = (1/delta)*(-E*delta1+F*delta2-G*delta3)
        # print("MY, NY := ", MY, NY)
        delta1 = 4*(XR*YU-XU*YR)
        delta2 = 4*(XL*YU-XU*YL)
        delta3 = 4*(XL*YR-XR*YL)
        MZ = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NZ = (1/delta)*(E*delta1-F*delta2+G*delta3)
        
        a = MX*MX+MY*MY+MZ*MZ - 1
        b = 2*(MX*NX+MY*NY+MZ*NZ)
        c = NX*NX+NY*NY+NZ*NZ

        if (b*b-4*a*c) < 0:
            x = np.NaN
            y = np.NaN
            z = np.NaN
            return x, y, z

        k1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
        k2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
        
        x1 = MX*k1+NX+bs1.x
        y1 = MY*k1+NY+bs1.y
        z1 = MZ*k1+NZ+bs1.z
        x2 = MX*k2+NX+bs1.x
        y2 = MY*k2+NY+bs1.y
        z2 = MZ*k2+NZ+bs1.z
        # print('        %.6f, %.6f' % (z1, z2))

        if k2 < 0:
            x = x1
            y = y1
            z = z1
        else:
            r_ref = math.sqrt((x1-bs1.x)**2+(y1-bs1.y)**2+(z1-bs1.z)**2)
            r2_ref = math.sqrt((x1-bs2.x)**2+(y1-bs2.y)**2+(z1-bs2.z)**2)
            r3_ref = math.sqrt((x1-bs3.x)**2+(y1-bs3.y)**2+(z1-bs3.z)**2)
            r4_ref = math.sqrt((x1-bs4.x)**2+(y1-bs4.y)**2+(z1-bs4.z)**2)
            sum1 = ((r2_ref - r_ref) - L)**2 + ((r3_ref - r_ref) - R)**2 + ((r4_ref - r_ref) - U)**2
            
            r_ref2 = math.sqrt((x2-bs1.x)**2+(y2-bs1.y)**2+(z2-bs1.z)**2)
            r2_ref2 = math.sqrt((x2-bs2.x)**2+(y2-bs2.y)**2+(z2-bs2.z)**2)
            r3_ref2 = math.sqrt((x2-bs3.x)**2+(y2-bs3.y)**2+(z2-bs3.z)**2)
            r4_ref2 = math.sqrt((x2-bs4.x)**2+(y2-bs4.y)**2+(z2-bs4.z)**2)
            sum2 = ((r2_ref2 - r_ref2) - L)**2 + ((r3_ref2 - r_ref2) - R)**2 + ((r4_ref2 - r_ref2) - U)**2
            # print('sum1, sum2 := %.6e %.6e (%.6f %.6f %.6f) (%.6f %.6f %.6f)' % (sum1, sum2, x1, y1, z1, x2, y2, z2))

            if sum1 < sum2 and (x1 > 0) and (y1 > 0) and (z1>2) and (z1 < 100):
                x = x1
                y = y1
                z = z1
            else:
                x = x2
                y = y2
                z = z2
        return x, y, z
    
    def scipy_3d_solver():
        if method.lower() == 'newton':
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=2000)
        elif method.lower() == 'taylor-direct':
            # print("Use the taylor-direct method ...")
            r21 = light_speed * dt21
            r31 = light_speed * dt31
            r41 = light_speed * dt41
            x_est, y_est, z_est = taylor_solver(bs1, bs2, bs3, bs4, r21, r31, r41)
        else:
            x = leastsq(equations_3d, (x_init, y_init, z_init))
            x_est = x[0][0]
            y_est = x[0][1]
            z_est = x[0][2]
        # print("solver() results : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        if (np.isnan(x_est) or np.isnan(y_est) or np.isnan(z_est)):
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=1000)
            # print("solver() results (through modified) : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        return x_est, y_est, z_est
    x_est, y_est, z_est = scipy_3d_solver()
    position.x = x_est
    position.y = y_est
    position.z = z_est
    return position

def tdoa_positioning_5bs_assist(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51, x_init, y_init, z_init, method='newton'):
    position = Sim3DCord(0.0, 0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    def equations_3d(p):
        x, y, z = p
        r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
        r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
        r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
        r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
        return (r2 - r1 - light_speed*dt21, r3 - r1 - light_speed*dt31, r4 - r1 - light_speed*dt41)
    
    def taylor_solver(bs1, bs2, bs3, bs4, bs5, L, R, U, r51):
        x = 0
        y = 0
        z = 0
        XL = bs2.x - bs1.x
        YL = bs2.y - bs1.y
        ZL = bs2.z - bs1.z
        XR = bs3.x - bs1.x
        YR = bs3.y - bs1.y
        ZR = bs3.z - bs1.z
        XU = bs4.x - bs1.x
        YU = bs4.y - bs1.y
        ZU = bs4.z - bs1.z
        E = L*L - XL*XL - YL*YL - ZL*ZL
        F = R*R - XR*XR - YR*YR - ZR*ZR
        G = U*U - XU*XU - YU*YU - ZU*ZU
        delta = -8 * (XL*YR*ZU+XU*YL*ZR+XR*YU*ZL-XL*YU*ZR-XR*YL*ZU-XU*YR*ZL)
        delta1 = 4*(YR*ZU-YU*ZR)
        delta2 = 4*(YL*ZU-YU*ZL)
        delta3 = 4*(YL*ZR-YR*ZL)
        # print("delta := ", delta, delta1, delta2, delta3)
        MX = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NX = (1/delta)*(E*delta1-F*delta2+G*delta3)
        # print("MX, NX := ", MX, NX)
        delta1 = 4*(XR*ZU-XU*ZR)
        delta2 = 4*(XL*ZU-XU*ZL)
        delta3 = 4*(XL*ZR-XR*ZL)
        MY = (2/delta)*(-L*delta1+R*delta2-U*delta3)
        NY = (1/delta)*(-E*delta1+F*delta2-G*delta3)
        # print("MY, NY := ", MY, NY)
        delta1 = 4*(XR*YU-XU*YR)
        delta2 = 4*(XL*YU-XU*YL)
        delta3 = 4*(XL*YR-XR*YL)
        MZ = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NZ = (1/delta)*(E*delta1-F*delta2+G*delta3)
        
        a = MX*MX+MY*MY+MZ*MZ - 1
        b = 2*(MX*NX+MY*NY+MZ*NZ)
        c = NX*NX+NY*NY+NZ*NZ
        
        if (b*b-4*a*c) < 0:
            x = np.NaN
            y = np.NaN
            z = np.NaN
            return x, y, z

        k1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
        k2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
        
        x1 = MX*k1+NX+bs1.x
        y1 = MY*k1+NY+bs1.y
        z1 = MZ*k1+NZ+bs1.z
        x2 = MX*k2+NX+bs1.x
        y2 = MY*k2+NY+bs1.y
        z2 = MZ*k2+NZ+bs1.z
        # print('        (%.6f %.6f %.6f), (%.6f %.6f %.6f)' % (x1, y1, z1, x2, y2, z2))
        
        if k2 < 0:
            x = x1
            y = y1
            z = z1
        else:
            r_ref = math.sqrt((x1-bs1.x)**2+(y1-bs1.y)**2+(z1-bs1.z)**2)
            r2_ref = math.sqrt((x1-bs2.x)**2+(y1-bs2.y)**2+(z1-bs2.z)**2)
            r3_ref = math.sqrt((x1-bs3.x)**2+(y1-bs3.y)**2+(z1-bs3.z)**2)
            r4_ref = math.sqrt((x1-bs4.x)**2+(y1-bs4.y)**2+(z1-bs4.z)**2)
            sum1 = ((r2_ref - r_ref) - L)**2 + ((r3_ref - r_ref) - R)**2 + ((r4_ref - r_ref) - U)**2
            
            r_ref2 = math.sqrt((x2-bs1.x)**2+(y2-bs1.y)**2+(z2-bs1.z)**2)
            r2_ref2 = math.sqrt((x2-bs2.x)**2+(y2-bs2.y)**2+(z2-bs2.z)**2)
            r3_ref2 = math.sqrt((x2-bs3.x)**2+(y2-bs3.y)**2+(z2-bs3.z)**2)
            r4_ref2 = math.sqrt((x2-bs4.x)**2+(y2-bs4.y)**2+(z2-bs4.z)**2)
            sum2 = ((r2_ref2 - r_ref2) - L)**2 + ((r3_ref2 - r_ref2) - R)**2 + ((r4_ref2 - r_ref2) - U)**2

            if sum1 < 1E-6 and sum2 < 1E-6:
                r5_ref = math.sqrt((x1-bs5.x)**2+(y1-bs5.y)**2+(z1-bs5.z)**2)
                r5_ref2 = math.sqrt((x2-bs5.x)**2+(y2-bs5.y)**2+(z2-bs5.z)**2)
                assist1 = (r5_ref - r_ref - r51)**2
                assist2 = (r5_ref2 - r_ref2 - r51)**2
                # print('assist1, assist2 := %.6e %.6e (%.6f %.6f %.6f) (%.6f %.6f %.6f)' % (assist1, assist2, x1, y1, z1, x2, y2, z2))
                if assist1 < assist2:
                    x = x1
                    y = y1
                    z = z1
                else:
                    x = x2
                    y = y2
                    z = z2
            else:
                if (sum1 < sum2) and (x1 > 0) and (y1 > 0) and (z1>2):
                    x = x1
                    y = y1
                    z = z1
                else:
                    x = x2
                    y = y2
                    z = z2
        return x, y, z
    
    def scipy_3d_solver():
        if method.lower() == 'newton':
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=2000)
        elif method.lower() == 'taylor-direct':
            # print("Use the taylor-direct method ...")
            r21 = light_speed * dt21
            r31 = light_speed * dt31
            r41 = light_speed * dt41
            r51 = light_speed * dt51
            x_est, y_est, z_est = taylor_solver(bs1, bs2, bs3, bs4, bs5, r21, r31, r41, r51)
        else:
            x = leastsq(equations_3d, (x_init, y_init, z_init))
            x_est = x[0][0]
            y_est = x[0][1]
            z_est = x[0][2]
        if (np.isnan(x_est) or np.isnan(y_est) or np.isnan(z_est)):
            x = leastsq(equations_3d, (x_init, y_init, z_init))
            x_est = x[0][0]
            y_est = x[0][1]
            z_est = x[0][2]
            # print("solver() results (through modified) : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        print("solver() results : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        return x_est, y_est, z_est
    x_est, y_est, z_est = scipy_3d_solver()
    position.x = x_est
    position.y = y_est
    position.z = z_est
    return position

