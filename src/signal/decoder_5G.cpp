// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "signal/decoder_5G.h"

using namespace std;

int8_t uci_decoder_1bit_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm)
{
    int8_t value_decoded[2] = {0, 1};
    int8_t decbits = 0;
    uint32_t n_size = qm;

    if (uci_bit != 1) {
        fprintf(stderr, "wrong input for uci_decoder_1bit_decoding() \n");
        return 0;
    }

    // Rate recovery
#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    int8_t rec[n_size] = {0};
    int8_t soft_enc[n_size][2] = {0};
#else
    int8_t *rec = new int8_t[n_size];
    int8_t *soft_enc[] = new int8_t*[n_size];
    for (int i = 0; i < n_size; i++) {
        soft_enc[i] = new int8_t[2];
        soft_enc[i][0] = 0; soft_enc[i][1] = 0;
    }
#endif
    float distmet[2] = {0};
    float sum1[2] = {0};
    float sum2[2] = {0};
    uint32_t match_len = coded_bit_len >= n_size ? n_size : coded_bit_len;
    for (uint32_t i = 0; i < match_len; i++)
    {
        rec[i] = llr[i];
    }

    // Fill the distances
    if (qm != 1) {
        for (int i = 0; i < 2; i++)
        {
            int8_t bit = value_decoded[i];
            uint32_t offset = 0;
            soft_enc[offset++][i] = bit;
            soft_enc[offset++][i] = bit;    // Placeholder -2
            for (int8_t k = 0; k < (qm - 2); k++) {
                soft_enc[offset+k][i] = 1;  // Placeholder -1
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        for (uint32_t k = 0; k < n_size; k++) {
            soft_enc[k][i] = 1 - 2 * soft_enc[k][i];
        }
    }
    // Calculate the euclide distance
    for (int i = 0; i < 2; i++) {
        for (uint32_t k = 0; k < n_size; k++) {
            float m = float(soft_enc[k][i]);
            float tmp = float(rec[k] - m);
            sum1[i] += fabs(tmp) * fabs(tmp);
            sum2[i] += fabs(m) * fabs(m);
        }
        distmet[i] = sum1[i] / sum2[i];
    }

#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    (void)0;
#else
    delete []rec;
    for (int i = 0; i < n_size; i++) {
        delete soft_enc[i];
    }
    delete []soft_enc;
#endif

    // Return the value
    if (distmet[0] <= distmet[1]) {
        return 0;
    } else {
        return 1;
    }
}

// Decoded results
// ===============
// 00 : return 0
// 01 : return 1
// 10 : return 2
// 11 : return 3
int8_t uci_decoder_2bits_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm)
{
    int8_t value_decoded[4][2] = {0, 0, 1, 0, 0, 1, 1, 1};
    int8_t decbits = 0;
    uint32_t n_size = 3 * qm;

    if (uci_bit != 2) {
        fprintf(stderr, "wrong input for uci_decoder_2bits_decoding() \n");
        return 0;
    }

    // Rate recovery
#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    int8_t rec[n_size] = {0};
    int8_t soft_enc[n_size][4] = {0};
#else
    int8_t *rec = new int8_t[n_size];
    int8_t *soft_enc[] = new int8_t*[n_size];
    for (int i = 0; i < n_size; i++) {
        soft_enc[i] = new int8_t[4];
        soft_enc[i][0] = 0; soft_enc[i][1] = 0;
        soft_enc[i][2] = 0; soft_enc[i][3] = 0;
    }
#endif
    float distmet[4] = {0};
    float sum1[4] = {0};
    float sum2[4] = {0};
    uint32_t match_len = coded_bit_len >= n_size ? n_size : coded_bit_len;
    for (uint32_t i = 0; i < match_len; i++) {
        rec[i] = llr[i];
    }

    // Fill the distances
    for (int i = 0; i < 4; i++)
    {
        int8_t bit1 = value_decoded[i][0];
        int8_t bit2 = value_decoded[i][1];
        int8_t c2 = bit1 ^ bit2;
        if (qm == 1) {
            soft_enc[0][i] = bit1;
            soft_enc[1][i] = bit2;
            soft_enc[2][i] = c2;
        } else {
            uint32_t offset = 0;
            soft_enc[offset++][i] = bit1;
            soft_enc[offset++][i] = bit2;
            for (int k = 0; k < (qm-2); k++)  soft_enc[offset++][i] = 1;    // Placeholder -1
            soft_enc[offset++][i] = c2;
            soft_enc[offset++][i] = bit1;
            for (int k = 0; k < (qm-2); k++)  soft_enc[offset++][i] = 1;    // Placeholder -1
            soft_enc[offset++][i] = bit2;
            soft_enc[offset++][i] = c2;
            for (int k = 0; k < (qm-2); k++)  soft_enc[offset++][i] = 1;    // Placeholder -1
        }
    }

    for (int i = 0; i < 4; i++) {
        for (uint32_t k = 0; k < n_size; k++) {
            soft_enc[k][i] = 1 - 2 * soft_enc[k][i];
        }
    }
    // Calculate the euclide distance
    for (int i = 0; i < 4; i++) {
        for (uint32_t k = 0; k < n_size; k++) {
            float m = float(soft_enc[k][i]);
            float tmp = float(rec[k] - m);
            sum1[i] += fabs(tmp) * fabs(tmp);
            sum2[i] += fabs(m) * fabs(m);
        }
        distmet[i] = sum1[i] / sum2[i];
    }
    float min_dist = distmet[0];
    int8_t pos;
    for (int8_t i = 1; i < 4; i++)
    {
        if (distmet[i] < min_dist)
        {
            pos = i;
        }
    }
#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    (void)0;
#else
    delete []rec;
    for (int i = 0; i < n_size; i++) {
        delete soft_enc[i];
    }
    delete []soft_enc;
#endif

    return pos;
}

uint32_t get_3gpp_n(uint32_t K, uint32_t E, uint32_t n_max)
{
    uint32_t n1, r_min, n_min, n2, n;

    if ((E <= (9.0/8)*pow(2, (ceil(log2(E))-1))) && ((double)K/E < 9.0/16))
    {
        n1 = ceil(log2(E)) - 1;
    }
    else
    {
        n1 = ceil(log2(E));
    }

    r_min = 1.0/8;
    n_min = 5;
    n2 = ceil(log2(K/r_min));
    // Find the minimum of (n1, n2, n_max), then find the bigger between the result and n_min
    n = std::max(n_min, std::min(std::min(n1, n2), n_max));

    return pow(2, n);
}

void polar_decoder_decoding(int8_t *a_hat, int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t list_size)
{
    uint32_t c = 0, p, p2;
    uint32_t K, E_r, N;
    // D^6+D^5+1
    int8_t crc_polynomial_pattern1[7] = {1, 1, 0, 0, 0, 0, 1};
    // D^11+D^10+D^9+D^5+1
    int8_t crc_polynomial_pattern2[12] = {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    int8_t *crc_polynomial_pattern = NULL;

    if (uci_bit < 12) {
        fprintf(stderr, "polar_3gpp un-supported block length, A should be no less than 12 !\n");
        return;
    } else if (uci_bit > 1706) {
        fprintf(stderr, "polar_3gpp un-supported block length, A should be no greater than 1706 !\n");
        return;
    } else if (uci_bit <= 19) {
        // Use PCCA-Polar
        c = 1;
        crc_polynomial_pattern = crc_polynomial_pattern1;
        p = 6;
    } else {
        // Use CA-Polar
        crc_polynomial_pattern = crc_polynomial_pattern2;
        if ((uci_bit >= 360 && coded_bit_len >= 1088) || (uci_bit >= 1013)) {
            c = 2;
        } else {
            c = 1;
        }
        p = 11;
    }
    
    p2 = 3;
    // Determine the number of information and CRC bits
    K = ceil(uci_bit/c) + p;
    E_r = floor(coded_bit_len/c);

    if (E_r > 8192) {
        fprintf(stderr, "polar_3gpp unsupported block length, G is too long !\n");
    }

    // Determine the number of bits used at the input and output of the polar encoder kernel
    N = get_3gpp_n(K, E_r, 10);

    // Get a rate matching pattern

    // Get a sequence pattern

    // Get the channel interleaving pattern

    if (uci_bit <= 19) {
        // Use PCCA-polar
        // Perform channel interleaving

        // Use 3 PC bits

        // Get an information bit pattern

        // Get a PC bit pattern

        // Perform polar decoding
    } else {
        // Use CA-Polar
        // Get an information bit pattern

        if (c == 2)
        {
            // Perform channel interleaving for first segment

            // Perform polar decoding for first segment

        }
        else
        {
            // Perform channel interleaving

            // Perform polar decoding
        }
    }
}
