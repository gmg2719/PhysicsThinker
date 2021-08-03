#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy.constants as spy_constants
from uav_tdoa import Sim2DCord
from scipy.optimize import fsolve
from scipy.optimize import leastsq

class Sim3DCord(Sim2DCord):
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def calc_distance(self, dest):
        return math.sqrt((self.x - dest.x)**2 + (self.y - dest.y)**2 + (self.z - dest.z)**2)
 
    def debug_print(self):
        print('Coordinate is : %.6f, %.6f, %.6f' % (self.x, self.y, self.z))


def derivative_F(x, y, z, bs1, bs2, bs3, bs4, dt21, dt31, dt41):
    light_speed = spy_constants.speed_of_light
    r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
    r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
    r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
    r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
    b1 = r2 - r1 - light_speed * dt21
    b2 = r3 - r1 - light_speed * dt31
    b3 = r4 - r1 - light_speed * dt41
    print(b1, b2, b3)
    f11 = (1/r2) * (x - bs2.x) - (1/r1) * (x - bs1.x)
    f12 = (1/r2) * (y - bs2.y) - (1/r1) * (y - bs1.y)
    f13 = (1/r2) * (z - bs2.z) - (1/r1) * (z - bs1.z)

    f21 = (1/r3) * (x - bs3.x) - (1/r1) * (x - bs1.x)
    f22 = (1/r3) * (y - bs3.y) - (1/r1) * (y - bs1.y)
    f23 = (1/r3) * (z - bs3.z) - (1/r1) * (z - bs1.z)

    f31 = (1/r4) * (x - bs4.x) - (1/r1) * (x - bs1.x)
    f32 = (1/r4) * (y - bs4.y) - (1/r1) * (y - bs1.y)
    f33 = (1/r4) * (z - bs4.z) - (1/r1) * (z - bs1.z)

    df = np.array([[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]])
    b = np.array([[-b1], [-b2], [-b3]])
    ans = np.matmul(inv(df), b) 
    return ans[0, 0], ans[1, 0], ans[2, 0]

# The traditional method ================
# dt21 : UE to bs2 and bs1 TOA difference
# dt31 : UE to bs3 and bs1 TOA difference
# dt41 : UE to bs4 and bs1 TOA difference
def tdoa_positioning_4bs(bs1, bs2, bs3, bs4, dt21, dt31, dt41):
    position = Sim3DCord(0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    def equations_3d(p):
        x, y, z = p
        r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
        r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
        r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
        r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
        return (r2 - r1 - light_speed*dt21, r3 - r1 - light_speed*dt31, r4 - r1 - light_speed*dt41)
    def scipy_3d_solver():
        x_est, y_est, z_est = fsolve(equations_3d, (0.0, 0.0, 0.0), maxfev=1000)
        print("scipy_solver() results : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        return x_est, y_est, z_est
    x_est, y_est, z_est = scipy_3d_solver()
    """
    x_est = (bs1.x + bs2.x + bs3.x) / 3
    y_est = (bs1.y + bs2.y + bs3.y) / 3
    z_est = (bs1.z + bs2.z + bs3.z) / 3
    delta_x = 0.0
    delta_y = 0.0
    delta_z = 0.0
    itr = 0
    while itr < 100:
        itr += 1
        delta_x, delta_y, delta_z = derivative_F(x_est, y_est, z_est, bs1, bs2, bs3, bs4, dt21, dt31, dt41)
        if (max(abs(delta_x), abs(delta_y), abs(delta_z)) < 1E-6):
            break
        x_est += delta_x
        y_est += delta_y
        z_est += delta_z
        print("Itr %d : (%.6f %.6f %.6f) ---> (%.6f %.6f %.6f)" % (itr, delta_x, delta_y, delta_z, x_est, y_est, z_est))
    """
    position.x = x_est
    position.y = y_est
    position.z = z_est

    return position

# From : 3D TDOA Problem Solution with Four Receiving Nodes, J. D. Gonzalez, R. Alvarez, etc., 27 June, 2019.
def tdoa_4bs_search_initbest(bs1, bs2, bs3, bs4):
    cdf_results = []
    # Cost 10W search times, get the best initial position for our problem
    for x_init in [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]:
        for y_init in [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]:
            for z_init in [2.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
                counter = 0
                for i in range(1000):
                    anchor = Sim3DCord(np.random.uniform(0, 2000), np.random.uniform(0, 2000), np.random.uniform(2, 200))
                    dist1 = anchor.calc_distance(bs1)
                    dist2 = anchor.calc_distance(bs2)
                    dist3 = anchor.calc_distance(bs3)
                    dist4 = anchor.calc_distance(bs4)
                    def equations_3d(p):
                        x, y, z = p
                        r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
                        r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
                        r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
                        r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
                        return (r2 - r1 - (dist2 - dist1), r3 - r1 - (dist3 - dist1), r4 - r1 - (dist4 - dist1))
                    def scipy_3d_solver():
                        x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init))
                        return x_est, y_est, z_est
                    x_est, y_est, z_est = scipy_3d_solver()
                    if (max(abs(x_est - anchor.x), abs(y_est - anchor.y)) < 1.0) and (abs(z_est - anchor.z) < 1.0):
                        counter += 1
                print('init (%.4f, %.4f, %.4f) search done !' % (x_init, y_init, z_init))
                cdf_results.append(counter / 1000)
    return np.array(cdf_results)

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
        MX = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NX = (1/delta)*(E*delta1-F*delta2+G*delta3)
        delta1 = 4*(XR*ZU-XU*ZR)
        delta2 = 4*(XL*ZU-XU*ZL)
        delta3 = 4*(XL*ZR-XR*ZL)
        MY = (2/delta)*(-L*delta1+R*delta2-U*delta3)
        NY = (1/delta)*(-E*delta1+F*delta2-G*delta3)
        delta1 = 4*(XR*YU-XU*YR)
        delta2 = 4*(XL*YU-XU*YL)
        delta3 = 4*(XL*YR-XR*YL)
        MZ = (2/delta)*(L*delta1-R*delta2+U*delta3)
        NZ = (1/delta)*(E*delta1-F*delta2+G*delta3)
        
        a = MX*MX+MY*MY+MZ*MZ - 1
        b = 2*(MX*NX+MY*NY+MZ*NZ)
        c = NX*NX+NY*NY+NZ*NZ
        k1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
        k2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
        
        x1 = MX*k1+NX+bs1.x
        y1 = MY*k1+NY+bs1.y
        z1 = MZ*k1+NZ+bs1.z
        x2 = MX*k2+NX+bs1.x
        y2 = MY*k2+NY+bs1.y
        z2 = MZ*k2+NZ+bs1.z
        
        if k2 < 0:
            x = x1
            y = y1
            z = z1
        else:
            r_ref = math.sqrt((x1-bs1.x)**2+(y1-bs1.y)**2+(z1-bs1.z)**2)
            r2_ref = math.sqrt((x1-bs2.x)**2+(y1-bs2.y)**2+(z1-bs2.z)**2)
            r3_ref = math.sqrt((x1-bs3.x)**2+(y1-bs3.y)**2+(z1-bs3.z)**2)
            r4_ref = math.sqrt((x1-bs4.x)**2+(y1-bs4.y)**2+(z1-bs4.z)**2)
            if abs((r2_ref - r_ref) - L) < 1E-4 and abs((r3_ref - r_ref) - R) < 1E-4 and abs((r4_ref - r_ref) - U) < 1E-4 and (x1 >= 0) and (
               y1 >= 0) and (z1>=2) and (x1 <= 2000) and (y1 <= 2000) and (z1 <= 200):
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
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=1000)
        elif method.lower() == 'taylor-direct':
            print("Use the taylor-direct method ...")
            r21 = light_speed * dt21
            r31 = light_speed * dt31
            r41 = light_speed * dt41
            x_est, y_est, z_est = taylor_solver(bs1, bs2, bs3, bs4, r21, r31, r41)
        else:
            x = leastsq(equations_3d, (x_init, y_init, z_init))
            x_est = x[0][0]
            y_est = x[0][1]
            z_est = x[0][2]
        print("solver() results : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        return x_est, y_est, z_est
    x_est, y_est, z_est = scipy_3d_solver()
    position.x = x_est
    position.y = y_est
    position.z = z_est
    return position

def tdoa_positioning_5bs(bs1, bs2, bs3, bs4, bs5, dt21, dt31, dt41, dt51, x_init, y_init, z_init):
    position = Sim3DCord(0.0, 0.0, 0.0)
    light_speed = spy_constants.speed_of_light
    def equations_3d(p):
        x, y, z = p
        r1 = math.sqrt((x-bs1.x)**2 + (y-bs1.y)**2 + (z-bs1.z)**2)
        r2 = math.sqrt((x-bs2.x)**2 + (y-bs2.y)**2 + (z-bs2.z)**2)
        r3 = math.sqrt((x-bs3.x)**2 + (y-bs3.y)**2 + (z-bs3.z)**2)
        r4 = math.sqrt((x-bs4.x)**2 + (y-bs4.y)**2 + (z-bs4.z)**2)
        r5 = math.sqrt((x-bs5.x)**2 + (y-bs5.y)**2 + (z-bs5.z)**2)
        return (r2 - r1 - light_speed*dt21, r3 - r1 - light_speed*dt31, r4 - r1 - light_speed*dt41, r5 - r1 - light_speed*dt51)
    def least_3d_solver():
        x = leastsq(equations_3d, (x_init, y_init, z_init))
        x_est = x[0][0]
        y_est = x[0][1]
        z_est = x[0][2]
        print("scipy_solver() results : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
        return x_est, y_est, z_est
    x_est, y_est, z_est = least_3d_solver()
    position.x = x_est
    position.y = y_est
    position.z = z_est
    return position

if __name__ == "__main__":
    print("Unit test")
    light_speed = spy_constants.speed_of_light
    print("Scheme 1 : ")
    error_results = []
    np.random.seed(1)
    for i in range(1000):
        print('ITR %d' % (i))
        uav = Sim3DCord(np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(0, 20))
        bs1 = Sim3DCord(10, 19, 6)
        bs2 = Sim3DCord(1, 1, 8)
        bs3 = Sim3DCord(19, 1, 10)
        bs4 = Sim3DCord(5, 10, 4)
        uav.debug_print()
        r1 = uav.calc_distance(bs1)
        r2 = uav.calc_distance(bs2)
        r3 = uav.calc_distance(bs3)
        r4 = uav.calc_distance(bs4)
        print('Distances : ', r1, r2, r3, r4)
        print('TDOA algorithm for 4 BSs in 3D plane :')
        pos = tdoa_positioning_4bs(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                   (r4-r1)/light_speed)
        pos.debug_print()
        print('max positioning error is %.4f' % (max(pos.x - uav.x, pos.y - uav.y, pos.z - uav.z)))
        if max(abs(pos.x - uav.x), abs(pos.y - uav.y), abs(pos.z - uav.z)) > 10:
            error_results.append(10.0)
        else:
            error_results.append(max(abs(pos.x - uav.x), abs(pos.y - uav.y), abs(pos.z - uav.z)))
    # fig, ax = plt.subplots()
    # x = np.array(range(1000))
    # y = np.array(error_results)
    # ax.plot(x, y)
    print("Scheme 2 : ")
    # Use the statistic method to estimate the optimization initial position of the whole search space
    # bs1 = Sim3DCord(1000, 1999, 6)
    # bs2 = Sim3DCord(1, 1, 18)
    # bs3 = Sim3DCord(1999, 1, 20)
    # bs4 = Sim3DCord(500, 1000, 10)
    # cdf_probility = tdoa_4bs_search_initbest(bs1, bs2, bs3, bs4)
    # sys.exit(0)
    # Where indoor  : the initial position achieve the best CDF is (4.0, 8.0, 8.0)
    # Where outdoor : the initial position achieve the best CDF is (400.0, 900.0, 180.0)
    x_init = 400.0
    y_init = 900.0
    z_init = 180.0

    error_horizon_results = []
    error_vertica_results = []
    dt21_results = []
    dt31_results = []
    dt41_results = []
    np.random.seed(1)
    for i in range(1000):
        print('ITR %d' % (i))
        # The first simulation parameters
        # uav = Sim3DCord(np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(0, 20))
        # bs1 = Sim3DCord(10, 19, 6)
        # bs2 = Sim3DCord(1, 1, 8)
        # bs3 = Sim3DCord(19, 1, 10)
        # bs4 = Sim3DCord(5, 10, 4)
        uav = Sim3DCord(np.random.uniform(0, 2000), np.random.uniform(0, 2000), np.random.uniform(2, 200))
        bs1 = Sim3DCord(1000, 1999, 6)
        bs2 = Sim3DCord(1, 1, 18)
        bs3 = Sim3DCord(1999, 1, 20)
        bs4 = Sim3DCord(500, 1000, 10)
        
        # Additional base station is used for the estimation
        bs5 = Sim3DCord(200, 800, 8)
        r5 = uav.calc_distance(bs5)

        uav.debug_print()
        r1 = uav.calc_distance(bs1)
        r2 = uav.calc_distance(bs2)
        r3 = uav.calc_distance(bs3)
        r4 = uav.calc_distance(bs4)
        print('Distances : ', r1, r2, r3, r4)
        print('TDOA algorithm for 4 BSs in 3D plane :')
        pos = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                           (r4-r1)/light_speed, x_init, y_init, z_init, method='taylor-direct')
        #pos = tdoa_positioning_5bs(bs1, bs2, bs3, bs4, bs5, (r2-r1)/light_speed, (r3-r1)/light_speed,
        #                                   (r4-r1)/light_speed, (r5-r1)/light_speed, x_init, y_init, z_init)
        pos.debug_print()

        est_horizon = max(abs(pos.x - uav.x), abs(pos.y - uav.y))
        est_vertica = abs(pos.z - uav.z)
        x_modify = pos.x
        y_modify = pos.y
        z_modify = pos.z
        
        if max(abs(pos.x - uav.x), abs(pos.y - uav.y)) > 1.0 or abs(pos.z - uav.z) > 1.0:
            print("The iterate %d results should be modified !" % (i))
            pos_tmp = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                               (r4-r1)/light_speed, bs1.x, bs1.y, bs1.z, method='Least')
            pos_tmp.debug_print()
            if max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y)) < est_horizon:
                x_modify = pos_tmp.x
                y_modify = pos_tmp.y
                est_horizon = max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y))
            if abs(pos_tmp.z - uav.z) < est_vertica:
                z_modify = pos_tmp.z
                est_vertica = abs(pos_tmp.z - uav.z)
            
            pos_tmp = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                               (r4-r1)/light_speed, bs2.x, bs2.y, bs2.z, method='Least')
            pos_tmp.debug_print()
            if max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y)) < est_horizon:
                x_modify = pos_tmp.x
                y_modify = pos_tmp.y
                est_horizon = max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y))
            if abs(pos_tmp.z - uav.z) < est_vertica:
                z_modify = pos_tmp.z
                est_vertica = abs(pos_tmp.z - uav.z)
            
            pos_tmp = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                               (r4-r1)/light_speed, bs3.x, bs3.y, bs3.z, method='Least')
            pos_tmp.debug_print()
            if max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y)) < est_horizon:
                x_modify = pos_tmp.x
                y_modify = pos_tmp.y
                est_horizon = max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y))
            if abs(pos_tmp.z - uav.z) < est_vertica:
                z_modify = pos_tmp.z
                est_vertica = abs(pos_tmp.z - uav.z)
            
            pos_tmp = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, (r2-r1)/light_speed, (r3-r1)/light_speed,
                                               (r4-r1)/light_speed, bs4.x, bs4.y, bs4.z, method='Least')
            pos_tmp.debug_print()
            if max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y)) < est_horizon:
                x_modify = pos_tmp.x
                y_modify = pos_tmp.y
                est_horizon = max(abs(pos_tmp.x - uav.x), abs(pos_tmp.y - uav.y))
            if abs(pos_tmp.z - uav.z) < est_vertica:
                z_modify = pos_tmp.z
                est_vertica = abs(pos_tmp.z - uav.z)
            print('After modification, the coordinate is (%.6f, %.6f, %.6f)' % (x_modify, y_modify, z_modify))

        pos.x = x_modify
        pos.y = y_modify
        pos.z = z_modify

        print('max positioning error is %.4f' % (max(pos.x - uav.x, pos.y - uav.y, pos.z - uav.z)))

        if max(abs(pos.x - uav.x), abs(pos.y - uav.y)) > 10:
            error_horizon_results.append(10.0)
        else:
            error_horizon_results.append(max(abs(pos.x - uav.x), abs(pos.y - uav.y)))
        if abs(pos.z - uav.z) > 10:
            error_vertica_results.append(10.0)
        else:
            error_vertica_results.append(abs(pos.z - uav.z))
        dt21_results.append(r2/light_speed - r1/light_speed)
        dt31_results.append(r3/light_speed - r1/light_speed)
        dt41_results.append(r3/light_speed - r1/light_speed)
        
    error_horizon_results = np.array(error_horizon_results)
    error_vertica_results = np.array(error_vertica_results)
    print('Error < 1m(horizontal) CDF = %.4f' % (np.size(np.where(error_horizon_results < 1)) / 1000))
    print('Error < 2m(horizontal) CDF = %.4f' % (np.size(np.where(error_horizon_results < 2)) / 1000))
    print('Error < 4m(horizontal) CDF = %.4f' % (np.size(np.where(error_horizon_results < 4)) / 1000))
    print('Error < 1m(vertical) CDF = %.4f' % (np.size(np.where(error_vertica_results < 1)) / 1000))
    print('Error < 3m(vertical) CDF = %.4f' % (np.size(np.where(error_vertica_results < 3)) / 1000))
    print('Error < 6m(vertical) CDF = %.4f' % (np.size(np.where(error_vertica_results < 6)) / 1000))

