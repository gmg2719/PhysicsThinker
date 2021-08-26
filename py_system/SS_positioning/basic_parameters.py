#!/usr/bin/python3
# -*- coding: utf-8 -*-

class SysParameters(object):
    """
    The parameters are demonstrated for the 5G wireless system.
    """
    def __init__(self):
        """Initilization operation."""
        self.nrb = 273
        self.subcarrier_spacing = 30
        self.nfft = 4096
        self.cyclic_prefix = 'normal'
        self.tx_ants = 2
        self.rx_ants = 2
        self.n_re_total = 3276
        self.n_symb_total = 14
        self.ul_slot_setting = [7, 8, 9, 17, 18, 19]

    def debug_print(self):
        print('SysParameters instance :')
        print('------------------------------------------')
        print('nrb = %d' % (self.nrb))
        print('subcarrier_spacing = %d' % (self.subcarrier_spacing))
        print('nfft = %d' % (self.nfft))
        print('cyclic_prefix = %s' % (self.cyclic_prefix))
        print('tx_ants = %d' % (self.tx_ants))
        print('rx_ants = %d' % (self.rx_ants))

class SrsPdu(object):
    """
    The parameters are from the external input file or just some settings.
    """
    def __init__(self):
        self.rnti = 1000
        self.handle = 0
        self.bwp_size = 273
        self.bwp_start = 0
        self.subcarrier_spacing = 1
        self.cyclic_prefix = 0
        self.num_ant_ports = 1
        self.num_symbols = 0
        self.num_repetitions = 0
        self.time_start_position = 13
        self.config_index = 0
        self.sequence_id = 0
        self.bandwidth_index = 0
        self.comb_size = 0
        self.comb_offset = 0
        self.cyclic_shift = 0
        self.frequency_position = 0
        self.frequency_shift = 0
        self.frequency_hopping = 0
        self.group_or_sequence_hopping = 0
        self.resource_type = 2
        self.t_srs = 1
        self.t_offset = 0

class SrsPduFormatted(object):
    """
    Store the SRS PDU necessary information used for UL positioning.
    """
    def __init__(self):
        self.rnti = 0
        self.num_ports = 0
        self.cyclic_shift = 0
        self.num_symbols = 0
        self.start_symbol_index = 0
        self.comb_size = 0
        self.comb_offset = 0
        self.config_index = 0
        self.bandwidth_index = 0
        self.frequency_position = 0
        self.frequency_shift = 0
        self.frequency_hopping = 0
        self.sequence_id = 0
        self.group_or_sequence_hopping = 0
        self.resource_type = 0
        self.t_srs = 0
        self.t_offset = 0

def srs_pdu2formatted(pdu, pdu_formatted):
        pdu_formatted.rnti = pdu.rnti
        pdu_formatted.num_ports = 2**pdu.num_ant_ports
        pdu_formatted.cyclic_shift = pdu.cyclic_shift
        pdu_formatted.num_symbols = 2**pdu.num_symbols
        pdu_formatted.start_symbol_index = pdu.time_start_position
        pdu_formatted.comb_size = 2**(pdu.comb_size + 1)
        pdu_formatted.comb_offset = pdu.comb_offset
        pdu_formatted.config_index = pdu.config_index
        pdu_formatted.bandwidth_index = pdu.bandwidth_index
        pdu_formatted.frequency_position = pdu.frequency_position
        pdu_formatted.frequency_shift = pdu.frequency_shift
        pdu_formatted.frequency_hopping = pdu.frequency_hopping
        pdu_formatted.sequence_id = pdu.sequence_id
        if (pdu.group_or_sequence_hopping == 0):
            pdu_formatted.group_or_sequence_hopping = 'neither'
        elif (pdu.group_or_sequence_hopping == 1):
            pdu_formatted.group_or_sequence_hopping = 'group_hopping'
        else:
            pdu_formatted.group_or_sequence_hopping = 'sequence_hopping'
        if (pdu.resource_type == 0) :
            pdu_formatted.resource_type = 'aperiodic'
        elif (pdu.resource_type == 1) :
            pdu_formatted.resource_type = 'semi-persistent'
        else:
            pdu_formatted.resource_type = 'periodic'
        pdu_formatted.t_srs = pdu.t_srs
        pdu_formatted.t_offset = pdu.t_offset

if __name__ == "__main__":
    print("Unit testing !")
    sys_parameters = SysParameters()
    sys_parameters.debug_print()

