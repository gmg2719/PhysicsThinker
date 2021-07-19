#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np

class polarDecode(object):
    def __init__(self, llr, crc_pattern):
        self.llr = llr
        self.crc_poly = crc_pattern
    
    def ca_polar_decode(self, mode, rate_match_pattern, info_bit, L, p2):
        e = self.llr.size
        n = info_bit.size
        
        if mode == 'repetition':
            if e < n:
                print('mode repetition is not compatible with e')
            else:
                d_tilde = np.zeros((1, n))
                for i in range(e):
                    d_tilde[0][int(rate_match_pattern[0][i])] = d_tilde[0][int(rate_match_pattern[0][i])] + self.llr[0][i]
        elif mode == 'puncturing':
            if e >= n:
                print('mode puncturing is not compatible with e')
            else:
                d_tilde = np.zeros((1, n))
                for i in range(e):
                    d_tilde[0][int(rate_match_pattern[0][i])] = self.llr[0][i]
        elif mode == 'shortening':
            if e >= n:
                print('mode shortening is not compatible with e')
            else:
                d_tilde = np.zeros((1, n))
                d_tilde[0][0:] = np.inf
                for i in range(e):
                    d_tilde[0][int(rate_match_pattern[0][i])] = self.llr[0][i]
        else:
            print('Un-supported type !')
        
        # Perform the SCL polar decode
        row_t = int(n)
        col_t = int(math.log2(n))
        zero = np.zeros((row_t, col_t))
        one = np.ones((row_t, 1))
        bits = np.zeros((row_t, col_t + 1))
        bit_update = np.c_[(1 - info_bit.T), zero]
        llrs = np.c_[zero, d_tilde.T]
        llrs_update = np.c_[zero, one]
        pm = np.array([0])
        l_prime = 0
        for i in range(n):
            self.update_llr(llrs, llrs_update, bits, bit_update, i, 0)
            if info_bit[0][i] == 0:
                if len(llrs.shape) == 2:
                    pm = self.phi(pm, llrs[i, 0], 0)
                else:
                    pm = self.phi(pm, llrs[i, 0, :], 0)
            else:
                if len(llrs.shape) == 2:
                    pm = np.c_[self.phi(pm, llrs[i, 0], 0), self.phi(pm, llrs[i, 0], 1)]
                    llrs = np.stack((llrs, llrs), axis=2)
                    bits = np.stack((bits, bits), axis=2)
                else:
                    pm = np.c_[self.phi(pm, llrs[i, 0, :], 0), self.phi(pm, llrs[i, 0, :], 1)]
                    llrs = np.concatenate((llrs, llrs), axis=2)
                    bits = np.concatenate((bits, bits), axis=2)
                bits[i, 0, 0:l_prime+1] = 0
                bits[i, 0, l_prime+1:2*(l_prime+1)] = 1
                bit_update[i, 0] = True
                
                l_prime = bits.shape[2]
                if l_prime > L:
                    max_indice = np.argsort(pm)
                    t = pm[0][max_indice[0][0]]
                    pm = np.array([t])
                    bits = bits[:,:,max_indice[0][0]]
                    llrs = llrs[:,:,max_indice[0][0]]
                    l_prime = L - 1
        # Information bit extraction
        a_hat = np.array([])
        p = self.crc_poly.size - 1
        max_indice = np.argsort(pm)
        length = min(L, 2**p2)
        for list_index in range(length):
            if len(bits.shape) == 2:
                u_hat = bits[:, 0].T
            else:
                u_hat = bits[:, 0, list_index].T
            i_t = np.where(info_bit == 1)[1]
            b_hat = u_hat[i_t]
            g_p = self.get_crc_generator_matrix(info_bit, self.crc_poly)
            item = np.dot(b_hat, g_p) % 2
            item2 = np.zeros((1, p), dtype=int)
            if p == np.sum(item == item2):
                a_hat = b_hat[0:-p]
                return a_hat
        return a_hat

    def update_llr(self, llrs, llrs_update, bit, bit_update, row, col):
        shape = bit.shape
        offset = int(shape[0] / 2**(shape[1]-col-1))
        if len(shape) == 2:
            if row % (2*offset) >= offset:
                if bit_update[row - offset, col] == False:
                    self.update_bit(bit, bit_update, row - offset, col)
                if llrs_update[row - offset, col + 1] == False:
                    self.update_llr(llrs, llrs_update, bit, bit_update, row - offset, col + 1)
                if llrs_update[row, col + 1] == False:
                    self.update_llr(llrs, llrs_update, bit, bit_update, row, col+1)
                llrs[row, col] = math.pow(-1, bit[row - offset, col]) * llrs[row - offset, col + 1] + llrs[row, col + 1]
            else:
                if llrs_update[row, col + 1] == False:
                    self.update_llr(llrs, llrs_update, bit, bit_update, row, col + 1)
                if llrs_update[row+offset, col+1] == False:
                    self.update_llr(llrs, llrs_update, bit, bit_update, row+offset, col+1)
                llrs[row, col] = self.min_star(llrs[row, col+1], llrs[row+offset, col+1])
        else:
            for i in range(shape[2]):
                if row % (2 * offset) >= offset:
                    if bit_update[row - offset, col] == False:
                        self.update_bit(bit, bit_update, row - offset, col)
                    if llrs_update[row - offset, col + 1] == False:
                        self.update_llr(llrs, llrs_update, bit, bit_update, row - offset, col + 1)
                    if llrs_update[row, col + 1] == False:
                        self.update_llr(llrs, llrs_update, bit, bit_update, row, col + 1)
                    llrs[row, col, i] = math.pow(-1, bit[row-offset, col]) * llrs[row-offset, col+1, i] + llrs[row, col+1, i]
                else:
                    if llrs_update[row, col+1] == False:
                        self.update_llr(llrs, llrs_update, bit, bit_update, row, col+1)
                    if llrs_update[row+offset, col+1] == False:
                        self.update_llr(llrs, llrs_update, bit, bit_update, row+offset, col+1)
                    llrs[row, col, i] = self.min_star(llrs[row, col+1, i], llrs[row+offset, col+1, i])
        llrs_update[row, col] = True
        return
    
    def update_bit(self, bit, bit_update, row, col):
        shape = bit.shape
        offset = int(shape[0] / 2**(shape[1] - col))
        if len(shape) == 2:
            if row % (2 * offset) >= offset:
                if bit_update[row, col - 1] == False:
                    self.update_bit(bit, bit_update, row, col-1)
                bit[row, col] = bit[row, col - 1]
            else:
                if bit_update[row, col - 1] == False:
                    self.update_bit(bit, bit_update, row, col - 1)
                if bit_update[row + offset, col - 1] == False:
                    self.update_bit(bit, bit_update, row+offset, col-1)
                bit[row, col] = (bit[row, col - 1] + bit[row + offset, col - 1]) % 2
        else:
            for i in range(shape[2]):
                if row % (2*offset) >= offset:
                    if bit_update[row, col-1] == False:
                        self.update_bit(bit, bit_update, row, col-1)
                    bit[row, col, i] = bit[row, col-1, i]
                else:
                    if bit_update[row, col-1] == False:
                        self.update_bit(bit, bit_update, row, col-1)
                    if bit_update[row+offset, col - 1] == False:
                        self.update_bit(bit, bit_update, row+offset, col-1)
                    bit[row, col, i] = (bit[row, col-1, i] + bit[row+offset, col-1, i]) % 2
        bit_update[row, col] = True
        return

    def min_star(self, a, b):
        return np.sign(a) * np.sign(b) * min(abs(a), abs(b))

    def phi(self, pm_minus1, li, ui):
        pm = np.copy(pm_minus1)
        flag = (0.5 * (1 - np.sign(li)) != ui)
        if np.sum(flag) > 0:
            if np.size(li) > 1:
                pm[0] += abs(li[0])
            else:
                pm[0] += abs(li)
        return pm

    def get_crc_generator_matrix(self, a, crc_poly):
        a = int(np.sum(a))
        p = crc_poly.size - 1
        if p < 1:
            print('some pattern wrong')
        g_p = np.zeros((a, p), dtype=int)
        if a > 0:
            g_p[a-1, :] = crc_poly[1:]
            list_t = np.arange(a-1, 0, -1)
            for k in list_t:
                item1 = np.r_[g_p[k, 1:], 0]
                item2 = g_p[k, 0] * crc_poly[1:]
                g_p[k-1, :] = item1 ^ item2
        return g_p
