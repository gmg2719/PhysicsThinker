#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_CURRENT_DIR)
from SysParameters import SysParameters

class WaveForm(object):
    """
    Provides dimensional information related to the OFDM modulation schemes. It 
    also returns extended numerology information about the 1ms slot structure 
    across the range of subcarrier spacings as well as window lengths for the 
    W-OFDM scheme. (The same as the MATLAB)
    Attributes:
        nrb: number of DL/UL resource blocks
        cyclic_prefix: optional for cyclic prefix, 'normal' or 'extended'
        windowing: used for CP-OFDM only, the number of time-domain samples
        subcarrier_spacing: subcarrier spacing (kHz)
        symbols_per_slot: 7 or 14 for normal CP, 12 for extended CP
        waveform_type: optional, 'CP-OFDM', 'W-OFDM', 'F-OFDM'
        alpha: W-OFDM only. Window roll-off factor
        sampling_rate: the sampling rate of the OFDM modulator
        nfft: the number of FFT points used in the OFDM modulator
        cyclic_prefix_lengths: cyclic prefix length of each OFDM symbol in a 1ms subframe
        symbol_lengths: symbol length of each OFDM symbol in a 1ms subframe
        nsubcarriers: number of subcarriers (nrb * 12)
        slots_per_subframe: number of slots per 1ms subframe
        symbols_per_subframe: number of symbols per 1ms subframe
        samples_per_subframe: number of samples per 1ms subframe
        subframe_period: subframe period (default 1ms)
        n1: W-OFDM only, number of samples in cyclic prefix
        n2: W-OFDM only, number of samples in cyclic suffix
    """
    def __init__(self):
        """Initilization operation."""
        self.nrb = 0
        self.cyclic_prefix = 'normal'
        self.windowing = 0
        self.subcarrier_spacing = 15   # kHz
        self.symbols_per_slot = 0
        self.waveform_type = 'CP-OFDM'
        self.alpha = 1.0
        self.sampling_rate = 0
        self.nfft = 0
        self.cyclic_prefix_lengths = []
        self.symbol_lengths = 0
        self.nsubcarriers = 0
        self.slots_per_subframe = 0
        self.symbols_per_subframe = 0
        self.samples_per_subframe = 0
        self.subframe_period = 0
        self.n1 = 0
        self.n2 = 0
    def _set_lte_numerology(self, enb):
        # Get baseline LTE CP-OFDM info
        tmp = math.log2(enb.nrb * 12/0.85)
        nfft = 2**(math.ceil(tmp))
        nfft = max(128, nfft)
        self.sampling_rate = nfft * 15E3
        self.nfft = int(nfft)
        ecp = (enb.cyclic_prefix.lower() == 'extended')
        w = 0
        if ecp == True:
            if nfft == 128:
                w = 4
            elif nfft == 256:
                w = 6
            elif nfft == 512:
                w = 4
            elif nfft == 1024:
                w = 6
            elif nfft == 2048:
                w = 8
        else:
            if nfft == 128:
                w = 4
            elif nfft == 256:
                w = 6
            elif nfft == 512:
                w = 4
            elif nfft == 1024:
                w = 6
            elif nfft == 2048:
                w = 8       
        if w == 0:
            # Additional rule for the other FFT sizes
            w = max(0, 8-2*(11-(math.log2(nfft))))
        self.windowing = w
        # CP lengths for 2048 point FFT per LTE slot (6 or 7 symbols)
        if ecp == True:
            cp_lengths = [512, 512, 512, 512, 512, 512]
        else:
            cp_lengths = [160, 144, 144, 144, 144, 144, 144]
        # Scale according to the FFT size and repeat the slots for a LTE subframe
        my_list = [int(i * float(nfft) / 2048) for i in cp_lengths]
        # The same as repeat the slots for a LTE subframe : repmat(my_list, 1, 2)
        self.cyclic_prefix_lengths = my_list + my_list

    def _set_nr_numerology(self, deltaf, slotduration):
        # 1ms subframe duration with 15kHz reference numerology
        # Validate and process subcarrier spacing
        log2df = math.log2(deltaf/15)
        scs_config = int(np.fix(log2df))
        if log2df < 0 or scs_config != log2df:
            print("The subcarrier spacing in kHz must equal 15*(2**n), and n must be a non-negative"
                  "integer in this function. Therefore valid values are 15,30,60,120,240 etc.", file=sys.stderr)
            sys.exit()
        # Scaling factor from the 15kHz reference
        scs_scale = 2**scs_config
        # scale OFDM sampling rate according to the subcarrier spacing
        self.sampling_rate = self.nfft * deltaf * 1000
        self.subcarrier_spacing = deltaf
        self.symbols_per_slot = slotduration
        self.slots_per_subframe = int(np.fix(14/slotduration)) * scs_scale
        self.symbols_per_subframe = self.symbols_per_slot * self.slots_per_subframe
        # Scale the cyclic prefix lengths, relative to the 15kHz reference, where,
        # for normal CP, the first OFDM symbol per 0.5ms (half subframe) will be longer
        xcp =  np.ones(self.symbols_per_subframe, dtype = np.int16)
        xcp = [self.cyclic_prefix_lengths[-1] * e for e in xcp]
        # fill the position 0 and len(xcp)/2
        xcp[0] = scs_scale * self.cyclic_prefix_lengths[0] - (scs_scale - 1) * self.cyclic_prefix_lengths[1]
        xcp[int(len(xcp)/2)] = xcp[0]
        self.cyclic_prefix_lengths = xcp
        self.symbol_lengths = [(i + self.nfft) for i in xcp]
        self.samples_per_subframe = sum(self.cyclic_prefix_lengths) + self.symbols_per_subframe * self.nfft
        self.subframe_period = float(self.samples_per_subframe) / self.sampling_rate
    def set_info(self, sys_parameters):
        # call private method
        self._set_lte_numerology(sys_parameters)
        # Get parameters which affect the extended functionality
        scs = sys_parameters.subcarrier_spacing
        symbolsperslot = len(self.cyclic_prefix_lengths)
        self.symbol_lengths = [ (e + self.nfft) for e in self.cyclic_prefix_lengths]
        self.nsubcarriers = sys_parameters.nrb * 12
        self._set_nr_numerology(scs, symbolsperslot)
        # Calculate the CP/CS lengths for W-OFDM, not supported currently
    def debug_print(self):
        print('WaveForm instance :')
        print('------------------------------------------')
        print('nrb = ', self.nrb)
        print('cyclic_prefix = %s' %(self.cyclic_prefix))
        print('windowing = ', self.windowing)
        print('subcarrier_spacing = ', self.subcarrier_spacing)
        print('symbols_per_slot = ', self.symbols_per_slot)
        print('waveform_type = %s' %(self.waveform_type))
        print('alpha = ', self.alpha)
        print('sampling_rate = ', self.sampling_rate)
        print('nfft = ', self.nfft)
        print('cyclic_prefix_lengths = ', self.cyclic_prefix_lengths)
        print('symbol_lengths = ', self.symbol_lengths)
        print('nsubcarriers = ', self.nsubcarriers)
        print('slots_per_subframe = ', self.slots_per_subframe)
        print('symbols_per_subframe = ', self.symbols_per_subframe)
        print('samples_per_subframe = ', self.samples_per_subframe)
        print('subframe_period = ', self.subframe_period)
        print('n1 = ', self.n1)
        print('n2 = ', self.n2)

if __name__ == "__main__":
    print("Unit testing !")
    sys_parameters = SysParameters()
    waveform = WaveForm()
    sys_parameters.nrb = 273
    sys_parameters.subcarrier_spacing = 30
    sys_parameters.nfft = 4096
    sys_parameters.cyclic_prefix = 'normal'
    sys_parameters.tx_ants = 2
    sys_parameters.rx_ants = 2
    waveform.set_info(sys_parameters)    
    waveform.debug_print()
