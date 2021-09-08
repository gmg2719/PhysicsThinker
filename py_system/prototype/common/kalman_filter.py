#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Kalman filter example demo in Python

import numpy as np

dt21 = [-2.7916670805284015e-06, -2.7418519459723104e-06, -2.8138314283679017e-06, \
        -2.7192640603865905e-06, -2.881320288591931e-06, -2.8042233137456083e-06,  \
        -2.7557836702075864e-06, -2.8172068212676725e-06, -2.7636655899422025e-06, -2.776973494522835e-06]

dt31 = [-3.035320791129759e-06, -2.9815469234812916e-06, -3.0597990034476893e-06, \
        -2.95705326710902e-06, -3.132218119530644e-06, -3.049481888405725e-06, \
        -2.996347562507334e-06, -3.06310238569571e-06, -3.00568974795052e-06, -3.019228426175292e-06]

dt41 = [-1.732617523443164e-06, -1.7007535302542273e-06, -1.7454480438910638e-06, \
        -1.6865517616508011e-06, -1.7897324256460051e-06, -1.739182244265951e-06, \
        -1.710245849076764e-06, -1.748459446026086e-06, -1.7131988002187747e-06, -1.7237848956490266e-06]

dt51 = [-2.6781525184987105e-06, -2.630203827974058e-06, -2.6992603457092162e-06, \
        -2.608506517280413e-06, -2.7644004303184575e-06, -2.6899904837008753e-06, \
        -2.643711056518053e-06, -2.7026495247696805e-06, -2.650953929781503e-06, -2.6641041505952245e-06]

def kalman_filter(n_iter, z):
    sz = (n_iter, )
    Q = 1E-5
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0.1**2 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

xhat21 = kalman_filter(10, dt21)
xhat31 = kalman_filter(10, dt31)
xhat41 = kalman_filter(10, dt41)
xhat51 = kalman_filter(10, dt51)
print(xhat21[9])
print(xhat31[9])
print(xhat41[9])
print(xhat51[9])
