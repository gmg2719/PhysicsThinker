#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
from numpy.linalg import inv
from scipy.optimize import fsolve

_X0 = 1000.0
_Y0 = 1999.0
_A1 = -0.9889300396346599
_A2 = -0.6110111606089454
_B1 = 2207.2860879738914
_B2 = 1367.4624639174315

def equations_A(x, y):
    r1_x = x - _X0
    r1_y = y - _Y0
    r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
    return _A1 * r1 + _B1, _A2 * r1 + _B2

def derivative_F(x, y):
    b1, b2 = equations((x, y))
    r1_x = x - _X0
    r1_y = y - _Y0
    r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
    f11 = _A1 * (1.0 / r1) * r1_x - 1.0
    f12 = _A1 * (1.0 / r1) * r1_y
    f21 = _A2 * (1.0 / r1) * r1_x
    f22 = _A2 * (1.0 / r1) * r1_y - 1.0
    df = np.array([[f11, f12], [f21, f22]])
    b = np.array([[-b1], [-b2]])
    ans = np.matmul(inv(df), b) 
    return ans[0, 0], ans[1, 0]

def equations(p):
    x0, x1 = p
    r1_x = x0 - _X0
    r1_y = x1 - _Y0
    r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
    return (_A1 * r1 + _B1 - x0, _A2 * r1 + _B2 - x1)

def scipy_solver():
    x_est, y_est = fsolve(equations, (0.0, 0.0))
    print("scipy_solver() results : (%.6f, %.6f)" % (x_est, y_est))
    return x_est, y_est

def fixedpoint_iterative_solver(total_iters):
    x_est = 0.0
    y_est = 0.0
    itr = 0
    while itr < total_iters:
        itr += 1
        x_est_before = x_est
        y_est_before = y_est
        x_est, y_est = equations_A(x_est, y_est)
        if (abs(x_est-x_est_before) < 1E-6) and (abs(y_est-y_est_before) < 1E-6):
            break
    print("Itr %d : fixedpoint_iterative_solver() TDOA results (%.6f, %.6f)" % (itr, x_est, y_est))
    return x_est, y_est

def newton_iterative_solver(total_iters):
    x_est = 0.0
    y_est = 0.0
    delta_x = 0.0
    delta_y = 0.0
    itr = 0
    while itr < total_iters:
        itr += 1
        delta_x, delta_y = derivative_F(x_est, y_est)
        if (max(abs(delta_x), abs(delta_y)) < 1E-6):
            break
        x_est += delta_x
        y_est += delta_y
        print("Itr %d : (%.6f %.6f) ---> (%.6f %.6f)" % (itr, delta_x, delta_y, x_est, y_est))
    print("Itr %d : newton_iterative_solver() TDOA results (%.6f, %.6f)" % (itr, x_est, y_est))
    return x_est, y_est

def direct_solver():
    x = 0.0
    y = 0.0
    alpha = _A2 / _A1
    beta = -(_A2/_A1)*_B1 + _B2
    A = (_A1**2) * (1 + alpha**2) - 1.0
    B = (_A1**2) * (2*alpha*beta-2*alpha*_Y0-2*_X0) + 2*_B1
    C = (_A1**2) * (_X0**2+beta**2+_Y0**2-2*_Y0*beta) - _B1**2
    print('Middle result is %.6f' % (B*B-4*A*C))
    x1 = (-B + np.sqrt(B*B-4*A*C))/(2*A)
    x2 = (-B - np.sqrt(B*B-4*A*C))/(2*A)
    y1 = alpha * x1 + beta
    y2 = alpha * x2 + beta

    if (x1 > 0) and (y1 > 0) and (x1 < 20) and (y1 < 20):
        x = x1
        y = y1
    else:
        x = x2
        y = y2
    print("direct_solver() TDOA results (%.6f, %.6f)" % (x, y))
    return x, y

if __name__ == "__main__":
    print("Unit test")
    scipy_solver()
    fixedpoint_iterative_solver(100)
    fixedpoint_iterative_solver(1000)
    newton_iterative_solver(10)
    direct_solver()
