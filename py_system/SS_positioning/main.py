#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from basic_parameters import SysParameters, SrsPdu, SrsPduFormatted, srs_pdu2formatted
from srs_app_scheduler import srs_app_positioning_schedule
from srs_app_scheduler import srs_app_run_indoors_sampling, srs_app_run_outdoors_sampling
from srs_app_scheduler import srs_app_singleUe_indoors, srs_app_singleUe_outdoors
from srs_app_scheduler import srs_app_multipleUe_indoors, srs_app_multipleUe_outdoors
from faked_app_scheduler import faked_app_singleUe_outdoors, faked_app_run_outdoors_sampling

print('Run simulation of the simplified contents !')
pdu = SrsPdu()
sys_parameters = SysParameters()
print("*********************************************")
print("*                                           *")
print("*              Set the parameters           *")
print("*                                           *")
print("*********************************************")
sys_parameters.tx_ants = 1
sys_parameters.rx_ants = 1
# PDU is extracted from the TTI SCF config information of the real 5G system
pdu.rnti = 51378
pdu.handle = 0
pdu.bwp_size = 273
pdu.bwp_start = 0
pdu.subcarrier_spacing = 1
pdu.cyclic_prefix = 0
pdu.num_ant_ports = 0
pdu.num_symbols = 0
pdu.num_repetitions = 0
pdu.time_start_position = 13
pdu.config_index = 63
pdu.sequence_id = 0
pdu.bandwidth_index = 0
pdu.comb_size = 1
pdu.comb_offset = 0
pdu.cyclic_shift = 0
pdu.frequency_position = 0
pdu.frequency_shift = 0
pdu.frequency_hopping = 0
pdu.group_or_sequence_hopping = 0
pdu.resource_type = 2
pdu.t_srs = 1
pdu.t_offset = 0

# The input parameters are formatted through the contents of 3GPP specification
srs_pdu = SrsPduFormatted()
srs_pdu2formatted(pdu, srs_pdu)

print('Start to run SRS signal send and receive')
print('AWGN channel model (indoors | outdoors)')
print("*********************************************")
print("*              Tx_power = 80/120 dB         *")
print("*          mode = 0, distance = 10.0m       *")
print("*          mode = 1, distance = 650.0m      *")
print("*             Run the simulation            *")
print("*********************************************")
np.random.seed(10)

#
# 1. UL-SRS-TDOA Positioning
#
srs_app_run_outdoors_sampling(sys_parameters, srs_pdu, 120, 1136.3806972916097, 1)
t = np.loadtxt('ta_outdoor_estimation.txt')
# srs_app_singleUe_outdoors(sys_parameters, srs_pdu, 120, 'ta')

#
# 2. Faked Positioning Test
#
# faked_app_singleUe_outdoors('kalman')
