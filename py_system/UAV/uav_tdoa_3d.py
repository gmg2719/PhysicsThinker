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
        # print("MZ, NZ := ", MZ, NZ)
        
        a = MX*MX+MY*MY+MZ*MZ - 1
        b = 2*(MX*NX+MY*NY+MZ*NZ)
        c = NX*NX+NY*NY+NZ*NZ
        # print('b = ', b)
        # print('a = ', a)
        # print('c = ', c)
        # print('b*b - 4*a*c = ', b*b-4*a*c)

        k1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
        k2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
        
        x1 = MX*k1+NX+bs1.x
        y1 = MY*k1+NY+bs1.y
        z1 = MZ*k1+NZ+bs1.z
        x2 = MX*k2+NX+bs1.x
        y2 = MY*k2+NY+bs1.y
        z2 = MZ*k2+NZ+bs1.z
        # print(x1, y1, z1)
        # print(x2, y2, z2)
        # print(k1, k2)
        
        if k2 < 0:
            x = x1
            y = y1
            z = z1
        else:
            r_ref = math.sqrt((x1-bs1.x)**2+(y1-bs1.y)**2+(z1-bs1.z)**2)
            r2_ref = math.sqrt((x1-bs2.x)**2+(y1-bs2.y)**2+(z1-bs2.z)**2)
            r3_ref = math.sqrt((x1-bs3.x)**2+(y1-bs3.y)**2+(z1-bs3.z)**2)
            r4_ref = math.sqrt((x1-bs4.x)**2+(y1-bs4.y)**2+(z1-bs4.z)**2)
            
            r_ref2 = math.sqrt((x2-bs1.x)**2+(y2-bs1.y)**2+(z2-bs1.z)**2)
            r2_ref2 = math.sqrt((x2-bs2.x)**2+(y2-bs2.y)**2+(z2-bs2.z)**2)
            r3_ref2 = math.sqrt((x2-bs3.x)**2+(y2-bs3.y)**2+(z2-bs3.z)**2)
            r4_ref2 = math.sqrt((x2-bs4.x)**2+(y2-bs4.y)**2+(z2-bs4.z)**2)
            
            # print(L, R, U)
            # print(r2_ref - r_ref, r3_ref - r_ref, r4_ref - r_ref)
            # print(r2_ref2 - r_ref2, r3_ref2 - r_ref2, r4_ref2 - r_ref2)
            
            # print("delta_t")
            # print(dt21, dt31, dt41)
            
            
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
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=2000)
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
        if (np.isnan(x_est) or np.isnan(y_est) or np.isnan(z_est)):
            x_est, y_est, z_est = fsolve(equations_3d, (x_init, y_init, z_init), maxfev=1000)
            print("solver() results (through modified) : (%.6f, %.6f, %.6f)" % (x_est, y_est, z_est))
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
    print("Scheme 2 : ")
    uav = Sim3DCord(88.4781, 85.2571, 73.9887)
    bs1 = Sim3DCord(1000, 1900, 60)
    bs2 = Sim3DCord(1000, 1000, 80)
    bs3 = Sim3DCord(1900, 1000, 10)
    bs4 = Sim3DCord(500, 1000, 40)
    
    t1 = [6.77627E-6, 6.77142E-6, 6.77686E-6, 6.76906E-6, 6.77653E-6]
    t2 = [4.30828E-6, 4.30719E-6, 4.30665E-6, 4.30742E-6, 4.30677E-6]
    t3 = [6.76459E-6, 6.76655E-6, 6.77671E-6, 6.76724E-6, 6.76873E-6]
    t4 = [3.34703E-6, 3.3495E-6, 3.34597E-6, 3.34808E-6, 3.34808E-6]

    uav.debug_print()
    r1 = uav.calc_distance(bs1)
    r2 = uav.calc_distance(bs2)
    r3 = uav.calc_distance(bs3)
    r4 = uav.calc_distance(bs4)
    print('Distances : ', r1, r2, r3, r4)
    print('Distance difference : ', r2 - r1, r3 - r1, r4 - r1)
    print('TDOA algorithm for 4 BSs in 3D plane :')
    print('TOA : ', r1/light_speed, r2/light_speed, r3/light_speed, r4/light_speed)
    print('TDOA : ', r2/light_speed - r1/light_speed, r3/light_speed - r1/light_speed, r4/light_speed - r1/light_speed)
    for k in [0, 1, 2, 3, 4]:
        print("===============================", k)
        pos = tdoa_positioning_4bs_improve(bs1, bs2, bs3, bs4, t2[k] - t1[k], t3[k] - t1[k], t4[k] - t1[k], 
                                           80, 80, 74, method='taylor-direct')
        print('AFTER Distance : ', pos.calc_distance(bs1), pos.calc_distance(bs2), pos.calc_distance(bs3), pos.calc_distance(bs4))
        print("===============================")

