#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import cmath
import numpy as np

from srs_receiver import SRS_BANDWIDTH_CONFIG_TABLE
from basic_sequences import LOWPAPRs, basic_generate_c_sequence, basic_cinit_calc

def srs_generate_u_v(group_or_seq_hopping, n_id, n_sf_u, offset, mzc):
    c_init = n_id
    if group_or_seq_hopping == 'neither':
        fgh = 0
        v = 0
    elif group_or_seq_hopping == 'group_hopping':
        c_len = 8 * (14 * n_sf_u + offset) + 8 + 1
        c_seq = basic_generate_c_sequence(c_init, c_len)
        c_seq = c_seq[-8:]
        fgh = basic_cinit_calc(c_seq, 8)
        v = 0
    elif group_or_seq_hopping == 'sequence_hopping':
        fgh = 0
        if mzc >= 72:
            c_len = 14 * n_sf_u + offset
            c_seq = basic_generate_c_sequence(c_init, c_len)
            v = c_seq[-1]
        else:
            v = 0
    else:
        print('Parameters of SRS group_or_seq_hopping error !')
        return 0, 0
    u = (fgh + n_id) % 30
    return u, v

def srs_freq_position_calc(srs_pdu):
    c_srs = srs_pdu.config_index
    b_srs = srs_pdu.bandwidth_index
    comb_size = srs_pdu.comb_size
    comb_offset = srs_pdu.comb_offset
    n_shift = srs_pdu.frequency_shift
    n_rrc = srs_pdu.frequency_position
    n_rb_sc = 12
    m_srs = SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2 * b_srs + 1]
    N = SRS_BANDWIDTH_CONFIG_TABLE[c_srs][2 * b_srs + 2]
    k0_bar = n_shift * n_rb_sc + comb_offset
    k0 = k0_bar
    for b in range(b_srs):
        n_b = np.floor(4 * n_rrc / m_srs) % N
        k0 += n_shift + m_srs * comb_size * n_b
    return m_srs, k0

def srs_tx_proc(sfn_id, slot_id, sys_parameters, srs_pdu):
    tx_grid = np.zeros((sys_parameters.n_re_total, sys_parameters.n_symb_total, sys_parameters.tx_ants), dtype=complex)
    
    if np.size(srs_pdu) != 1:
        print('Only support 1 SRS PDU right now !')
        sys.exit(-1)
    
    # Initialization
    num_rb, k0 = srs_freq_position_calc(srs_pdu)
    n_srs_re = int(num_rb * 12 / srs_pdu.comb_size)
    n_cs_max = 8 if srs_pdu.comb_size == 4 else 12
    baseseq = np.zeros((n_srs_re, srs_pdu.num_symbols), dtype=complex)
    
    # Generate base sequences
    sequence_id = srs_pdu.sequence_id
    group_or_sequence_hopping = srs_pdu.group_or_sequence_hopping
    for symb_idx in range(srs_pdu.num_symbols):
        offset = symb_idx + srs_pdu.start_symbol_index
        u, v = srs_generate_u_v(group_or_sequence_hopping, sequence_id, slot_id, offset, n_srs_re)
        low_p = LOWPAPRs(u, v, [], n_srs_re).gen_base()
        baseseq[:, symb_idx] = low_p
    print('SRS generate base sequences done !')

    # SRS resource mapping
    for symb_idx in range(srs_pdu.num_symbols):
        offset = symb_idx + srs_pdu.start_symbol_index
        for port_idx in range(srs_pdu.num_ports):
            cyclic_shift_port = (srs_pdu.cyclic_shift + np.floor(n_cs_max * port_idx / srs_pdu.num_ports)) % n_cs_max
            cyclic_shift_phase = np.zeros(n_srs_re, dtype=complex)
            for ii in range(n_srs_re):
                cyclic_shift_phase[ii] = cmath.exp(1j * 2 * cmath.pi * cyclic_shift_port / n_cs_max * ii)
            re_offset = int(k0)
            tx_grid[re_offset:re_offset+num_rb*12:srs_pdu.comb_size, offset, port_idx] = baseseq[:, symb_idx] * cyclic_shift_phase
    print('SRS resource mapping done !')

    return tx_grid

