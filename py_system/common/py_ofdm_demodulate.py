#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.append(_CURRENT_DIR)
sys.path.append(_ROOT_DIR)
import scipy as spy
from data.SysParameters import SysParameters
from data.WaveForm import WaveForm
from py_ofdm_modulate import ofdm_modulate

def ofdm_demodulate(sys_parameters, waveform_info, tx_wave):
    """
    Demodulate the OFDM signal
    Args:
        sys_parameters:
        waveform_info:
        tx_wave: the OFDM signal, 2D list, (samples, rxants)
    Returns
        the demodulated OFDM signal, 3D list, (subcarriers, symbols, rxants)
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
    # Use numpy.array to express the rxWaveform
    rxwave = np.array(tx_wave)
    rxant = np.shape(rxwave)[1]
    rx_grid = np.zeros((nsc, symbols_per_slot, rxant), dtype=complex)
    # Calculate position of the first active subcarrier in the FFT output,
    # according to the FFT size and the number of active subcarriers
    first_activesc = int((nfft - nsc) / 2)
    # Demodulate all symbols within the input data
    cpres = cp_lengths[0] - cp_lengths[1]
    # The same as the MATLAB operation : rxwave[0:cpres, :] = []
    rxwave = np.delete(rxwave, range(cpres), 0)
    rxwave = np.reshape(rxwave, (-1, symbols_per_slot, rxant), order="F")
    # The same as the MATLAB operation : rxwave[0:cp_lengths[1], :, :] = []
    rxwave = np.delete(rxwave, range(cp_lengths[1]), 0)
    for symbol in range(symbols_per_slot):
        for rr in range(rxant):
            in_list = rxwave[:, symbol, rr]
            # rxwave is the numpy.array, just div the value
            fft_output = spy.fftpack.fftshift(spy.fft.fft(in_list)) / math.sqrt(nfft)
            rx_grid[:, symbol, rr] = fft_output[first_activesc : (first_activesc + nsc)]
    return rx_grid

if __name__ == "__main__":
    print("Unit testing !")
    print("Prepare the OFDM generation")
    sys_parameters = SysParameters()
    waveform = WaveForm()
    sys_parameters.nrb = 273
    sys_parameters.subcarrier_spacing = 30
    sys_parameters.nfft = 4096
    sys_parameters.cyclic_prefix = 'normal'
    sys_parameters.tx_ants = 2
    sys_parameters.rx_ants = 2
    grid = np.zero((3276, 14, 2), dtype=complex)
    grid = grid + 1.0
    print(np.shape(grid))
    print("OFDM modulate")
    tx_wave = ofdm_modulate(sys_parameters, waveform, grid)
    print("Generate the signal done !")
    print(np.shape(tx_wave))
    print("Receive the OFDM signal, assume rx_value = tx_value")
    print("OFDM demodulate")
    rx_grid = ofdm_demodulate(sys_parameters, waveform, tx_wave)
    print("Demodulate the signal done !")
    print(np.shape(rx_grid))
