#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import scipy.constants as spy_constants
from scipy.optimize import fsolve

class Sim2DCord(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def debug_print(self):
        print('Coordinate is : ', self.x, self.y)

def calc_2D_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# The normal interface
def toa_positioning(bs, t):
    pass

def toa_positioning_3bs(bs1, bs2, bs3, t1, t2, t3):
    position = Sim2DCord(0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    r1 = t1 * light_speed
    r2 = t2 * light_speed
    r3 = t3 * light_speed
    x1 = bs1.x
    y1 = bs1.y
    x2 = bs2.x
    y2 = bs2.y
    x3 = bs3.x
    y3 = bs3.y
    gamma1 = x2*x2-x3*x3+y2*y2-y3*y3+r3*r3-r2*r2
    gamma2 = x1*x1-x2*x2+y1*y1-y2*y2+r2*r2-r1*r1
    
    position.x = ((y2-y1)*gamma1+(y2-y3)*gamma2)/((x2-x3)*(y2-y1)+(x1-x2)*(y2-y3))
    position.y = ((x2-x1)*gamma1+(x2-x3)*gamma2)/((x2-x1)*(y2-y3)+(x2-x3)*(y1-y2))
    position.x *= 0.5
    position.y *= 0.5
    
    return position

# The normal interface
def tdoa_positioning(bs_basic, bs, dt):
    pass

# dt1 : UE to bs2 and bs_basic TOA difference
# dt2 : UE to bs3 and bs_basic TOA difference
def tdoa_positioning_3bs(bs_basic, bs2, bs3, dt1, dt2, method='newton'):
    position = Sim2DCord(0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    tmp = np.array([[bs2.x - bs_basic.x, bs2.y - bs_basic.y], [bs3.x - bs_basic.x, bs3.y - bs_basic.y]])
    K1 = bs_basic.x**2 + bs_basic.y**2
    K2 = bs2.x**2 + bs2.y**2
    K3 = bs3.x**2 + bs3.y**2

    P1 = np.array((-1.0) * inv(tmp))
    r21 = light_speed*dt1
    r31 = light_speed*dt2
    P2 = np.array([[r21], [r31]])
    P3 = np.array([[0.5*(r21**2-K2+K1)], [0.5*(r31**2-K3+K1)]])
    A = np.matmul(P1, P2)
    B = np.matmul(P1, P3)
    x_est = 0.0
    y_est = 0.0
    
    def direct_solver():
        x0 = bs_basic.x
        y0 = bs_basic.y
        x = 0.0
        y = 0.0
        
        if (A[0, 0] == 0) and (A[1, 0] == 0):
            x = B[0, 0]
            y = B[1, 0]
            return x, y
        
        if (A[0, 0] == 0) and (A[1, 0] != 0):
            x = B[0, 0]
            a = A[1, 0]**2 - 1
            b = A[1, 0]**2*(-2*y0) + 2 * B[1, 0]
            c = A[1, 0]**2*((x-x0)**2 + y0**2) - B[1, 0]**2
            y1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
            y2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
            r_ref = math.sqrt((x-x0)**2+(y1-y0)**2)
            r2_ref = math.sqrt((x-bs2.x)**2+(y1-bs2.y)**2)
            r3_ref = math.sqrt((x-bs3.x)**2+(y1-bs3.y)**2)
            if abs((r2_ref - r_ref) - r21) < 1E-4 and abs((r3_ref - r_ref) - r31) < 1E-4 and (y1 >= 0) and (y1 <= 2000) and (y1 <= 2000):
                y = y1
            else:
                y = y2
            print("direct_solver() TDOA results (%.6f, %.6f)" % (x, y1))
            print("direct_solver() TDOA results (%.6f, %.6f)" % (x, y2))
            return x, y
        
        alpha = A[1, 0] / A[0, 0]
        beta = -(A[1, 0]/A[0, 0])*B[0, 0] + B[1, 0]
        a = (A[0, 0]**2) * (1 + alpha**2) - 1.0
        b = (A[0, 0]**2) * (2*alpha*beta-2*alpha*y0-2*x0) + 2*B[0, 0]
        c = (A[0, 0]**2) * (x0**2+beta**2+y0**2-2*y0*beta) - B[0, 0]**2
        print('Middle result is %.6f' % (b*b-4*a*c))
        x1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
        x2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
        y1 = alpha * x1 + beta
        y2 = alpha * x2 + beta

        r_ref = math.sqrt((x1-x0)**2+(y1-y0)**2)
        r2_ref = math.sqrt((x1-bs2.x)**2+(y1-bs2.y)**2)
        r3_ref = math.sqrt((x1-bs3.x)**2+(y1-bs3.y)**2)

        if abs((r2_ref - r_ref) - r21) < 1E-4 and abs((r3_ref - r_ref) - r31) < 1E-4 and (x1 >= 0) and (
           y1 >= 0) and (x1 <= 2000) and (y1 <= 2000):
            x = x1
            y = y1
        else:
            x = x2
            y = y2
        print("direct_solver() TDOA results (%.6f, %.6f)" % (x1, y1))
        print("direct_solver() TDOA results (%.6f, %.6f)" % (x2, y2))
        return x, y
    
    def equations(p):
        x0, x1 = p
        r1_x = x0 - bs_basic.x
        r1_y = x1 - bs_basic.y
        r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
        return (A[0, 0] * r1 + B[0, 0] - x0, A[1, 0] * r1 + B[1, 0] - x1)
    
    if method.lower() == 'direct':
        x_est, y_est = direct_solver()
    else:
        x_est, y_est = fsolve(equations, (0.0, 0.0))
    
    # Use Newton iterative method to estimate the non-linear system results
    # Iterate 100 times mostly
    """
    itr = 0
    while itr < 100:
        itr += 1
        x_est_before = x_est
        y_est_before = y_est
        r1_x = x_est - bs_basic.x
        r1_y = y_est - bs_basic.y
        r1 = math.sqrt(r1_x*r1_x + r1_y*r1_y)
        x_est = A[0, 0] * r1 + B[0, 0]
        y_est = A[1, 0] * r1 + B[1, 0]
        # print("Estimate(%d) : (%.8f %.8f)" % (itr, x_est, y_est))
        if (abs(x_est - x_est_before) < 1E-6) and (abs(y_est - y_est_before) < 1E-6):
            break
    """
    print("Estimate : ", x_est, y_est)
    position.x = x_est
    position.y = y_est

    return position

if __name__ == "__main__":
    print("Unit test")
    print(spy_constants.speed_of_light)     # Velocity magnitude is m/s
    light_speed = spy_constants.speed_of_light
    print("Scheme 2 : ")
    uav = Sim2DCord(1000, 1000)
    bs1 = Sim2DCord(1000, 1999)
    bs2 = Sim2DCord(1, 1)
    bs3 = Sim2DCord(1999, 1)
    uav.debug_print()
    r1 = calc_2D_distance(uav, bs1)
    print(r1)
    r2 = calc_2D_distance(uav, bs2)
    print(r2)
    r3 = calc_2D_distance(uav, bs3)
    print(r3)
    print('TOA algorithm for 3 BSs in 2D plane :')
    pos = toa_positioning_3bs(bs1, bs2, bs3, r1/light_speed, r2/light_speed, r3/light_speed)
    pos.debug_print()
    print('TDOA algorithm for 3 BSs in 2D plane :')
    pos = tdoa_positioning_3bs(bs1, bs2, bs3, r2/light_speed - r1/light_speed, r3/light_speed - r1/light_speed)
    pos.debug_print()
    print("Scheme 3:")
    error_results = []
    dt21_results = []
    dt31_results = []
    np.random.seed(1)
    for i in range(1000):
        print('ITR %d' % (i))
        uav = Sim2DCord(np.random.uniform(0, 2000), np.random.uniform(0, 2000))
        bs1 = Sim2DCord(1000, 1999)
        bs2 = Sim2DCord(1, 1)
        bs3 = Sim2DCord(1999, 1)
        uav.debug_print()
        r1 = calc_2D_distance(uav, bs1)
        print(r1)
        r2 = calc_2D_distance(uav, bs2)
        print(r2)
        r3 = calc_2D_distance(uav, bs3)
        print(r3)
        print('TOA algorithm for 3 BSs in 2D plane :')
        pos = toa_positioning_3bs(bs1, bs2, bs3, r1/light_speed, r2/light_speed, r3/light_speed)
        pos.debug_print()
        print('TDOA algorithm for 3 BSs in 2D plane :')
        dt21 = r2/light_speed - r1/light_speed + np.random.randn() * (8*1E-9)
        dt31 = r3/light_speed - r1/light_speed + np.random.randn() * (8*1E-9)
        pos = tdoa_positioning_3bs(bs1, bs2, bs3, dt21, dt31)

        print('max positioning error is %.4f' % (max(pos.x - uav.x, pos.y - uav.y)))
        if max(abs(pos.x - uav.x), abs(pos.y - uav.y)) > 10:
            error_results.append(10.0)
        else:
            error_results.append(max(abs(pos.x - uav.x), abs(pos.y - uav.y)))
        dt21_results.append(r2/light_speed - r1/light_speed)
        dt31_results.append(r3/light_speed - r1/light_speed)

    error_results = np.array(error_results)
    print('Error < 1m(horizontal) CDF = %.4f' % (np.size(np.where(error_results < 1)) / 1000))
    print('Error < 2m(horizontal) CDF = %.4f' % (np.size(np.where(error_results < 2)) / 1000))
    print('Error < 4m(horizontal) CDF = %.4f' % (np.size(np.where(error_results < 4)) / 1000))
    print('Error < 10m(horizontal) CDF = %.4f' % (np.size(np.where(error_results < 10)) / 1000))
    # fig, ax = plt.subplots()
    # x = np.array(range(1000))
    # y = np.array(error_results)
    # ax.plot(x, y)

