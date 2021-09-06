#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import numpy as np
from scipy.fft import fft, ifft
import scipy as spy

"""
# Original interface for the python programming
def iczt(x, m=None, w=None, a=None):
    # Translated from GNU Octave's czt.m
    n = len(x)
    if m is None: m = n
    if w is None: w = np.exp(-2j * np.pi / m)
    if a is None: a = 1
    
    chirp = w ** (np.arange(1 - n, max(m, n)) ** 2 / 2.0)
    N2 = int(2 ** np.ceil(np.log2(m + n - 1)))  # next power of 2
    xp = np.append(x * (1.0 / np.power(a, np.arange(n))) * chirp[n - 1 : n + n - 1], np.zeros(N2 - n))
    ichirpp = np.append(1 / chirp[: m + n - 1], np.zeros(N2 - (m + n - 1)))
    r = ifft(fft(xp) * fft(ichirpp))
    return r[n - 1 : m + n - 1] * chirp[n - 1 : m + n - 1]
"""

# Interface for the C/C++ programming
def czt(x, m=None, w=None, a=None):
    n = len(x)
    if m is None: m = n
    if w is None: w = np.exp(-2j * np.pi / m)
    if a is None: a = 1

    N2 = int(2 ** np.ceil(np.log2(2*n - 1)))  # next power of 2

    # For example, n = 24, the range is [-23, -22, ..., 22, 23]
    chirp = np.zeros(2*n-1, dtype=complex)
    ichirpp = np.zeros(N2, dtype=complex)
    index = 0
    for k in np.arange(1-n, n):
         chirp[index] = w ** (k ** 2 / 2.0)
         ichirpp[index] = 1 / chirp[index]
         index += 1

    xp = np.zeros(N2, dtype=complex)
    index = 0
    for k in np.arange(n-1, n+n-1):
        xp[index] = x[index] * chirp[k]
        index += 1

    r = ifft(fft(xp) * fft(ichirpp))
    
    index = 0
    result = np.zeros(n, dtype=complex)
    for k in np.arange(n-1, n+n-1):
        result[index] = r[k] * chirp[k]
        index += 1
    
    return result

def iczt(x, m=None, w=None, a=None):
    n = len(x)
    if m is None: m = n
    if w is None: w = np.exp(2j * np.pi / m)
    if a is None: a = 1
    
    N2 = int(2 ** np.ceil(np.log2(n)))  # next power of 2
    
    # For example, n = 24, the range is [-23, -22, ..., 22, 23]
    chirp = np.zeros(n, dtype=complex)
    ichirpp = np.zeros(N2, dtype=complex)
    index = 0
    for k in np.arange(1-n/2, 1+n/2):
         chirp[index] = w ** (k ** 2 / 2.0)
         ichirpp[index] = 1 / chirp[index]
         index += 1

    xp = np.zeros(N2, dtype=complex)
    index = 0
    for k in np.arange(0, n):
        xp[index] = x[index] * chirp[k]
        index += 1
    
    r = ifft(fft(xp) * fft(ichirpp))
    
    index = 0
    result = np.zeros(n, dtype=complex)
    for k in np.arange(0, n):
        result[index] = r[k] * chirp[k]
        index += 1
    
    return result/n
    
def cround(z, d=None):
    return np.round(z.real, d) + 1j * np.round(z.imag, d)

x16 = [7.62435794e-01 + 2.06383348e-01j, \
     1.95438373e+00 - 1.29716802e+00j, \
     -3.51395518e-01 + 2.51173091e+00j, \
     8.30021858e-01 + 2.47798109e+00j, \
     -8.85782421e-01 + 1.04149783e+00j, \
     -1.41291881e+00 + 2.89411402e+00j, \
     -1.00015211e+00 - 1.37304044e+00j, \
     -2.28566742e+00 - 6.59287274e-01j, \
     1.04745364e+00  + 7.48452485e-01j, \
     1.25504541e+00  - 4.69390452e-01j, \
     -4.25973117e-01 + 1.34006751e+00j, \
     1.77294597e-01  + 8.03263605e-01j, \
     -1.19099844e+00 + 3.62012446e-01j, \
     -1.95291626e+00 + 1.21275023e-01j, \
     1.28068149e+00  - 2.16396064e-01j, \
     -9.94455218e-01 - 1.08508790e+00j]

x24 = [7.62435794e-01 + 2.06383348e-01j, \
     1.95438373e+00 - 1.29716802e+00j, \
     -3.51395518e-01 + 2.51173091e+00j, \
     8.30021858e-01 + 2.47798109e+00j, \
     -8.85782421e-01 + 1.04149783e+00j, \
     -1.41291881e+00 + 2.89411402e+00j, \
     -1.00015211e+00 - 1.37304044e+00j, \
     -2.28566742e+00 - 6.59287274e-01j, \
     1.04745364e+00  + 7.48452485e-01j, \
     1.25504541e+00  - 4.69390452e-01j, \
     -4.25973117e-01 + 1.34006751e+00j, \
     1.77294597e-01  + 8.03263605e-01j, \
     -1.19099844e+00 + 3.62012446e-01j, \
     -1.95291626e+00 + 1.21275023e-01j, \
     1.28068149e+00  - 2.16396064e-01j, \
     -9.94455218e-01 - 1.08508790e+00j, \
    1.63691080e+00 + 1.24296121e-01j, \
    1.35439610e+00 - 2.50292659e+00j, \
    4.71289456e-02 + 1.99719679e+00j, \
    2.34237742e+00 + 1.72555804e+00j, \
    -1.30372810e+00 + 3.60458732e-01j, \
    -1.52314532e+00 + 1.17943203e+00j, \
    -6.24070354e-02 - 1.74195826e+00j, \
    -9.84873921e-02 - 1.50130713e+00j]

print(cround(iczt(x24), 6))
print(cround(ifft(x24), 6))
