#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
from math import gcd
import numpy as np
from scipy.fft import fft, ifft
import scipy as spy

def factorization(n):
    factors = []
    def get_factor(n):
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1
        while factor == 1:
            for count in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)
            cycle_size *= 2
            x_fixed = x
        return factor

    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next
    return factors

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
     
    if dif is None and dit is None :
        dif = False
        dit = True
    elif dit is None :
        dit = not dif
    else :
        dif = not dit
 
    if max_rad is None and min_rad is None :
        max_rad = False
        min_rad = True
    elif min_rad is None :
        min_rad = not max_rad
    else :
        max_rad = not min_rad
         
    N = len(x)
    plan = []
 
    if min_rad :
        fct = sorted(factorization(N))
        NN = N
        ptr = plan
        for f in fct :
            NN //= f
            P = [f, dft, None]
            Q = [NN, fft_CT, []]
            if dit :
                ptr += [Q, P]
            else :
                ptr += [P, Q]
            if NN == fct[-1] :
                Q[1:3] = [dft, None]
                break
            else :
                ptr = Q[2]
    else :
        items ={}
        def follow_tree(plan, N, fct, **kwargs) :
            if fct is None:
                fct = nrp_numth.naive_factorization(N)
            divs = sorted(nrp_numth.divisors(fct))
            val_P = divs[(len(divs) - 1) // 2]
            P = [val_P]
            if val_P in items :
                P = items[val_P]
            else :
                fct_P = nrp_numth.naive_factorization(val_P)
                if len(fct_P) > 1 :
                    P += [fft_CT, []]
                    follow_tree(P[2], val_P, fct_P, dit = dit)
                else :
                    P +=[dft, None]
                items[val_P] = P
            val_Q = N // val_P
            Q = [val_Q]
            if val_Q in items :
                Q = items[val_Q]
            else :
                fct_Q = nrp_numth.naive_factorization(val_Q)
                if len(fct_Q) > 1 :
                    Q += [fft_CT, []]
                    follow_tree(Q[2], val_Q, fct_Q, dit = dit)
                else :
                    Q +=[dft, None]
                items[val_Q] = Q
            if dit :
                plan += [Q, P]
            else :
                plan += [P, Q]
 
        follow_tree(plan, N, None, dit = dit)
 
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

def cround(z, d=None):
    return np.round(z.real, d) + 1j * np.round(z.imag, d)

np.random.seed(10)
x = np.random.rand(3) + np.random.rand(3) * 1j 
#plan = plan_fft(x)
#print(cround(fft_CT(x, plan), 6))
print(cround(ifft(x), 6))
    