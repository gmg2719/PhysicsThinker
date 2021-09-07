#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
from math import gcd
from itertools import count
import numpy as np
from scipy.fft import fft, ifft
import scipy as spy

POINTS =  np.array([6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632])
POINTS_FACTORS = np.array([
    # 6-48
    [3, 2],    # OK
    [3, 4],    # OK
    [3, 6] ,   # OK
    [3, 8] ,   # OK
    [5, 6] ,   # OK
    [3, 3, 4] , # OK
    [3, 16] ,  # OK
    #
    [3, 3, 6] , # OK
    [4, 3, 5] , # OK
    [3, 3, 8] , # OK
    
    
    [4, 3, 7] , # OK
    [6, 3, 5] , # OK
    [32, 3] ,  # OK
    [4, 3, 3, 3] , # OK
    [8, 3, 5] , # OK
    [4, 3, 11] , # OK
    [16, 3, 3] , # OK
    [6, 5, 5] , # OK
    [4, 3, 13] , # OK
    [2, 3, 3, 3, 3] , # OK
    [8, 3, 7] , # OK
    [4, 3, 3, 5],
    [64, 3] ,  # OK
    [4, 3, 17] , # OK
    [8, 3, 3, 3], # OK
    [4, 3, 19], # OK
    [3, 5, 16], # OK
    [4, 3, 3, 7], # OK
    [8, 3, 11] , #OK
    [6, 3, 3, 5] , #OK
    [4, 3, 23] , #OK
    [32, 3, 3] , #OK
    [4, 3, 5, 5] , #OK
    [8, 3, 13] , #OK
    [4, 3, 3, 3, 3] , #OK
    [16, 3, 7] , #OK
    [4, 3, 29] , #OK
    [8, 3, 3, 5] , #OK
    [128, 3] , # OK
    [4, 3, 3, 11] , #OK
    [8, 3, 17] ,
    [16, 3, 3, 3] ,
    [2, 3, 3, 5, 5] ,
    [8, 3, 19] ,
    [32, 3, 5] ,
    [2, 3, 3, 3, 3, 3] ,
    [8, 3, 3, 7] ,
    [16, 3, 11] ,
    [4, 3, 3, 3, 5] ,
    [8, 3, 23] ,
    [64, 3, 3] ,
    [8, 3, 5, 5] ,
    [16, 3, 13] ,
    [8, 3, 3, 3, 3] ,
    [32, 3, 7] ,
    [8, 3, 29] ,
    [16, 3, 3, 5] ,
    [2, 3, 5, 5, 5] ,
    [256, 3] ,
    [8, 3, 3, 11] ,
    [2, 3, 3, 3, 3, 5] ,
    [16, 3, 17] ,
    [32, 3, 3, 3] ,
    [4, 3, 3, 5, 5] ,
    [16, 3, 19] ,
    [64, 3, 5] ,
    [4, 3, 3, 3, 3, 3] ,
    [16, 3, 3, 7] ,
    [32, 3, 11] ,
    [8, 3, 3, 3, 5] ,
    [16, 3, 23] ,
    [128, 3, 3] ,
    [16, 3, 5, 5] ,
    [32, 3, 13] ,
    [16, 3, 3, 3, 3] ,
    [64, 3, 7] ,
    [2, 3, 3, 3, 5, 5] , #OK
    [32, 3, 3, 5] , #OK
    [2, 3, 3, 3, 3, 3, 3] ,
    [4, 3, 5, 5, 5] , #OK
    [512, 3] ,  # OK
    [16, 3, 3, 11] , #OK
    [4, 3, 3, 3, 3, 5] ,
    [32, 3, 17]]); #OK
"""
def factorization(n):
    factors=[]
    i = 2
    while n >= i:
        if n%i == 0:  
            factors.append(i) 
            n = n//i
            i=2 
        else:
            i = i+1 
    return factors

for k in range(np.size(POINTS)):
    print(factorization(POINTS[k]), ",")
sys.exit(0)
"""

def plan_fft(x, **kwargs) :
    """
    Returns an FFT plan for x.
 
    The plan uses direct DFT for prime sizes, and the generalized CT algorithm
    with either maximum or minimum radix for others.
 
    Keyword Arguments
     dif     -- If True, will use decimation in frequency. Default is False.
     dit     -- If True, will use decimation in time. Overrides dif. Default
                is True.
     max_rad -- If True, will use tje largest possible radix. Default is False.
     min_rad -- If True, will use smallest possible radix. Overrides max_rad.
                Default is True.
                
    """
    dif = kwargs.pop('dif', None)
    dit = kwargs.pop('dit', None)
    max_rad = kwargs.pop('max_rad', None)
    min_rad = kwargs.pop('min_rad', None)

    dif = False
    dit = True
    max_rad = False
    min_rad = True
         
    N = len(x)
    plan = []

    index = 0
    for k in range(84):
        if POINTS[k] == N:
            index = k
            break
    fct = POINTS_FACTORS[index]
    NN = N
    ptr = plan
    for f in fct :
        NN //= f
        P = [f, dft, None]
        Q = [NN, fft_CT, []]
        ptr += [Q, P]

        if NN == fct[-1] :
            Q[1:3] = [dft, None]
            break
        else :
            ptr = Q[2]
    print(plan[0])
    print(plan[1])
    return plan

def dft(x, **kwargs) :
    """
    Computes the (inverse) discrete Fourier transform naively.
 
    Arguments
     x         - Te list of values to transform
 
    Keyword Arguments
     inverse   - True to compute an inverse transform. Default is False.
     normalize - True to normalize inverse transforms. Default is same as
                 inverse.
     twiddles  - Dictionary of twiddle factors to be used by function, used
                 for compatibility as last step for FFT functions, don't set
                 manually!!!
 
    """
    inverse = kwargs.pop('inverse',False)
    normalize = kwargs.pop('normalize',inverse)
    N = len(x)
    inv = -1 if not inverse else 1
    twiddles = kwargs.pop('twiddles', {'first_N' : N, 0 : 1})
    NN = twiddles['first_N']
    X =[0] * N
    for k in range(N) :
        for n in range(0,N) :
            index = (n * k * NN // N) % NN
            twid= twiddles.setdefault(index,
                                      math.e**(inv*2j*math.pi*index/NN))
            X[k] += x[n] * twid
        if inverse and normalize :
             X[k] /= NN
    return X

def fft_CT(x, plan,**kwargs) :
    """
    Computes the (inverse) DFT using Cooley-Tukey's generalized algorithm.
     
    Arguments
     x        - The list of values to transform.
     plan     - A plan data structure for FFT execution.
 
    Keyword Arguments
     inverse  - if True, computes an inverse FFT. Default is False.
     twiddles - dictionary of twiddle factors to be used by function, used
                for ptimization during recursion, don't set manually!!!
                                 
    """
    inverse = kwargs.pop('inverse', False)
    inv = 1 if inverse else -1
    N = len(x)
    twiddles = kwargs.pop('twiddles', {'first_N' : N, 0 : 1})
    NN = twiddles['first_N']
    X = [0] * N
    P = plan[0]
    Q = plan[1]
    # do inner FFT
    for q in range(Q[0]) :
        X[q::Q[0]] = P[1](x[q::Q[0]], plan=P[2], twiddles=twiddles,
                          inverse=inverse,normalize=False)
    # multiply by twiddles
    for q in range(Q[0]) :
        for s in range(P[0]) :
            index = q * s * NN // N
             
            mult = twiddles.setdefault(index,
                                       math.e**(inv*2j*math.pi*index/NN))
            X[Q[0]*s+q] *= mult
    # do outer FFT
    for s in range(P[0]) :
        X[Q[0]*s:Q[0]*(s+1)] = Q[1](X[Q[0]*s:Q[0]*(s+1)],plan=Q[2],
                                    twiddles=twiddles,inverse=inverse,
                                    normalize=False)
    # transpose matrix
    ret = [0] * N
    for s in range(P[0]) :
        ret[s::P[0]] = X[s*Q[0]:(s+1)*Q[0]]
    if inverse and N == NN : # normalize inverse only in first call
        ret = [val / N for val in ret]
    
    return np.array(ret)

def fft_cooley_2factor(x, factors):
    P = factors[0]
    Q = factors[1]
    
    X1 = np.zeros((Q,P), dtype=complex)
    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = fft(X1[q])
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(-2j*math.pi*index/len(x))
            X1[q][s] *= mult

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = fft(tmp)
    return np.reshape(X1, (-1))

def fft_cooley_3factor(x, factors):
    P1 = factors[0]
    P2 = factors[1]
    Q = factors[2]
    X1 = np.zeros((Q, P1*P2), dtype=complex)

    P = P1 * P2
    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = fft_cooley_2factor(X1[q], [P1, P2])
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(-2j*math.pi*index/len(x))
            X1[q][s] *= mult

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = fft(tmp)
    return np.reshape(X1, (-1))

def fft_cooley_4factor(x, factors):
    P1 = factors[0]
    P2 = factors[1]
    P3 = factors[2]
    Q = factors[3]
    X1 = np.zeros((Q, P1*P2*P3), dtype=complex)
    
    P = P1 * P2 * P3
    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = fft_cooley_3factor(X1[q], [P1, P2, P3])
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(-2j*math.pi*index/len(x))
            X1[q][s] *= mult

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = fft(tmp)
    return np.reshape(X1, (-1))

def fft_cooley_5factor(x, factors):
    P1 = factors[0]
    P2 = factors[1]
    P3 = factors[2]
    P4 = factors[3]
    Q = factors[4]
    X1 = np.zeros((Q, P1*P2*P3*P4), dtype=complex)
    
    P = P1 * P2 * P3 * P4
    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = fft_cooley_3factor(X1[q], [P1, P2, P3, P4])
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(-2j*math.pi*index/len(x))
            X1[q][s] *= mult

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = fft(tmp)
    return np.reshape(X1, (-1))

def fft_cooley_recursion(x, factors):

    if (np.size(factors) == 2):
        return fft_cooley_2factor(x, factors)
    
    P_list = factors[0:-1]
    Q = factors[-1]
    P = int(np.prod(P_list))

    X1 = np.zeros((Q, P), dtype=complex)

    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = fft_cooley_recursion(X1[q], P_list)
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(-2j*math.pi*index/len(x))
            X1[q][s] *= mult

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = fft(tmp)
    return np.reshape(X1, (-1))

length_N = []

def ifft_cooley_2factor(x, factors):
    P = factors[0]
    Q = factors[1]
    
    X1 = np.zeros((Q,P), dtype=complex)
    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = ifft(X1[q])
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(2j*math.pi*index/len(x))
            if (index > len(x)):
                print("XXX")
            X1[q][s] *= mult
    length_N.append(len(x))
    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = ifft(tmp)
    return np.reshape(X1, (-1))

def ifft_cooley_recursion(x, factors):

    if (np.size(factors) == 2):
        return ifft_cooley_2factor(x, factors)
    
    P_list = factors[0:-1]
    Q = factors[-1]
    P = int(np.prod(P_list))

    X1 = np.zeros((Q, P), dtype=complex)

    for q in range(Q):
        for s in range(P):
            X1[q][s] = x[s*Q + q]
    # Step 1
    for q in range(Q):
        X1[q] = ifft_cooley_recursion(X1[q], P_list)
    
    # Step 2
    for q in range(Q):
        for s in range(P):
            index = q * s
            mult = math.e**(2j*math.pi*index/len(x))
            if (index > len(x)):
                print("XXX")
            X1[q][s] *= mult
    length_N.append(len(x))

    # Step 3
    for s in range(P):
        tmp = X1[:, s]
        X1[:, s] = ifft(tmp)
    return np.reshape(X1, (-1))

def cround(z, d=None):
    return np.round(z.real, d) + 1j * np.round(z.imag, d)

for k in range(84):
    points = POINTS[k]
    np.random.seed(10)
    x = np.random.rand(points) + np.random.rand(points) * 1j 
    res1 = cround(ifft_cooley_recursion(x, POINTS_FACTORS[k]), 6)
    res2 = cround(ifft(x), 6)
    if np.array_equal(res1, res2):
        print("[%d]Points %d is OK" % (k, points))
    else:
        print("[%d]Points %d is failure" % (k, points))
print(sorted(list(set(length_N))))
print(len(list(set(length_N))))
