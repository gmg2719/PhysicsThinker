#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.append(_ROOT_DIR)
import scipy.fftpack
import scipy as spy
from data.SysParameters import SysParameters
from data.WaveForm import WaveForm

def ofdm_modulate(sys_parameters, waveform_info, grid):
    """
    Generate OFDM modulate signal
    Args:
        sys_parameters :
        waveform_info :
        grid : frequency domain
    Returns
        the OFDM signal, time domain, 2D list, (samples, txants)
    Raises:
        None
    """
    # Get dimensionality information derived from the system parameters
    waveform_info.set_info(sys_parameters)
    # Cache the main dims
    nsc = waveform_info.nsubcarriers
    nfft = waveform_info.nfft
    cp_lengths = waveform_info.cyclic_prefix_lengths
    symbols_per_slot = waveform_info.symbols_per_slot
    cpsum = sum(cp_lengths[0:symbols_per_slot])
    nsamples = cpsum + symbols_per_slot * nfft
    # Get dimensional information derived from the resource grid
    # During the simulation, grid is 3D array : (subcarrier, symbol, rxant)
    nsymbols = np.shape(grid)[1]
    nants = np.shape(grid)[2]
    # Index of first subcarrier in IFFT input
    firstsc = int(nfft/2) - (sys_parameters.nrb * 6)
    tx_wave = np.zeros((nsamples, nants), dtype=complex)
    # Used for nants = 1 situation, Pingzhou Ming, 2021.5.26
    if nants == 1:    
        tx_wave = np.reshape(tx_wave, (nsamples, 1))
    # Modulate each OFDM symbol of the resource grid
    length = 0
    for i in range(nsymbols):
        fullstart = length
        ifftin = np.zeros((nfft, nants), dtype=complex)
        # Used for nants = 1 situation, Pingzhou Ming, 2021.5.26
        if nants == 1:    
            ifftin = np.reshape(ifftin, (nfft, 1))
        offset = int(nsc/2)
        ifftin[firstsc:(firstsc+offset),:] = grid[0:offset,i,:]
        ifftin[(firstsc + offset):(firstsc+2*offset),:] = grid[offset:,i,:]
        # Perform IFFT
        for tt in range(nants):
            in_list = ifftin[:,tt]
            # in_list is 1D list, iffout is the numpy.array, just mul the value
            iffout = spy.fft.ifft(spy.fftpack.fftshift(in_list)) * math.sqrt(nfft)
            # horizontal combination
            addcp = np.hstack((iffout[(nfft-cp_lengths[i]):nfft], iffout))
            if tt == 0:
                length = length + len(addcp)
            tx_wave[fullstart:length, tt] = addcp
    return tx_wave

if __name__ == "__main__":
    print("Unit testing !")
    print("Prepare the OFDM generation")
    sys_parameters = SysParameters()
    waveform = WaveForm()
    sys_parameters.nrb = 273
    sys_parameters.subcarrier_spacing = 30
    sys_parameters.nfft = 4096
    sys_parameters.cyclic_prefix = 'normal'
    sys_parameters.tx_ants = 1
    sys_parameters.rx_ants = 2
    grid = np.zero((3276, 14, 1), dtype=complex)
    grid = grid + 1.0
    print(np.shape(grid))
    print("OFDM modulate")
    tx_wave = ofdm_modulate(sys_parameters, waveform, grid)
    print("Generate the signal done !")
    print(np.shape(tx_wave))

