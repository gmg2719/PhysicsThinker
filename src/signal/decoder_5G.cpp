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

#include "signal/decoder_5G.h"

int8_t uci_decoder_1bit_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm)
{
    int8_t value_decoded[2] = {0, 1};
    int8_t decbits = 0;
    uint32_t n_size = qm;

    if (uci_bit != 1) {
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

    // Return the value
    if (distmet[0] <= distmet[1]) {
        return 0;
    } else {
        return 1;
    }

#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    delete []rec;
    for (int i = 0; i < n_size; i++) {
        delete soft_enc[i];
    }
    delete []soft_enc;
#endif
}

int8_t uci_decoder_2bits_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm)
{

}
