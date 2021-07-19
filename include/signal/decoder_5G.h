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

#ifndef _DECODER_5G_H_
#define _DECODER_5G_H_          1

#include <cstdint>

// These functions are extracted and re-design from the MATLAB Toolbox documents and some papers
// llr : likely-hood ration when receiver finish demodulate in wireless communication
// coded_bit_len : the length of llr represents the encode bits of the sender
// uci_bit : the bits of the sender, sometimes use the simple encoding method, 3GPP TS 38.212
// qm : the modulation order in wireless communication
int8_t uci_decoder_1bit_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm);
int8_t uci_decoder_2bits_decoding(int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t qm);

// Polar decoder for PUCCH and PUSCH of 3GPP NR, as defined in section 6.3 of TS 38.212. Implements 
// the code block segmentation and CRC attachment of sections 6.3.1.2.1 and 6.3.2.2.1, the channel 
// coding of sections 6.3.1.3.1 and 6.3.2.3.2, the rate matching of sections 6.3.1.4.1 and 6.3.2.4.1, 
// as well as the code block concatenation of sections 6.3.1.5.1 and 6.3.2.5.1.
// LLR : should be a real row vector comprising G number of Logarithmic Likelihood Ratios, each having
//       a value obtained as LLR=ln(P(bit=0)/P(bit=1)), where G should be no greater than 8192 if 
//       A<360 from sections 6.3.1.5 and 6.3.2.5 of TS 38.212, while the last LLR corresponds to g_G-1.
void polar_decoder_decoding(int8_t *a_hat, int8_t *llr, uint32_t coded_bit_len, uint32_t uci_bit, int8_t list_size);

#endif
