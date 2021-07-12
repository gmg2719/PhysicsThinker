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

#ifndef _MY_FFT_HPP_
#define _MY_FFT_HPP_            1

#include <cmath>
#include <utility>
#include "common/complex_t.h"
#include "fft_stockham_r4.hpp"

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_fft(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);
    const T theta0 = 2*M_PI/N;

    switch (N)
    {
    case 1 : {
        } return;
    case 2 : {
            // N = 2 is treated as the special situation
            const complex_t<T> a = x[0];
            const complex_t<T> b = x[1];
            x[0] = (a + b)/2;
            x[1] = (a - b)/2;
        } break;
    case 4 : {
            const complex_t<T> a = x[0];
            const complex_t<T> b = x[1];
            const complex_t<T> c = x[2];
            const complex_t<T> d = x[3];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[0] = (apc +  bpd)/4;
            x[1] = (amc - jbmd)/4;
            x[2] = (apc -  bpd)/4;
            x[3] = (amc + jbmd)/4;
        } break;
    case 8 : {
            // Pay attention, -sin(theta0)
            const complex_t<T> w1p = complex_t<T>(cos(theta0), -sin(theta0));
            const complex_t<T> w2p = w1p*w1p;
            const complex_t<T> w3p = w1p*w2p;
            complex_t<T> y[8];
            complex_t<T> a = x[0]; complex_t<T> b = x[2]; complex_t<T> c = x[4]; complex_t<T> d = x[6];
            complex_t<T>  apc =    a + c; complex_t<T>  amc =    a - c;
            complex_t<T>  bpd =    b + d; complex_t<T> jbmd = j*(b - d);
            y[0] = apc +  bpd; y[1] = amc - jbmd;
            y[2] = apc -  bpd; y[3] = amc + jbmd;
            a = x[1]; b = x[3]; c = x[5]; d = x[7];
            apc = a + c; amc = a - c;
            bpd = b + d; jbmd = j*(b - d);
            y[4] = apc +  bpd; y[5] = w1p * (amc - jbmd);
            y[6] = w2p * (apc -  bpd); y[7] = w3p * (amc + jbmd);
            a = y[0]; b = y[4];
            x[0] = (a + b)/8; x[4] = (a - b)/8;
            a = y[1]; b = y[5];
            x[1] = (a + b)/8; x[5] = (a - b)/8;
            a = y[2]; b = y[6];
            x[2] = (a + b)/8; x[6] = (a - b)/8;
            a = y[3]; b = y[7];
            x[3] = (a + b)/8; x[7] = (a - b)/8;
        } break;
    case 16: {
            complex_t<T> y[16];
            for (int p = 0; p < 4; p++) {
                // Pay attention, -sin(theta0)
                const complex_t<T> w1p = complex_t<T>(cos(p*theta0), -sin(p*theta0));
                const complex_t<T> w2p = w1p*w1p;
                const complex_t<T> w3p = w1p*w2p;

                const complex_t<T> a = x[p];
                const complex_t<T> b = x[p + 4];
                const complex_t<T> c = x[p + 8];
                const complex_t<T> d = x[p + 12];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[4*p] = apc +  bpd;
                y[4*p + 1] = w1p*(amc - jbmd);
                y[4*p + 2] = w2p*(apc -  bpd);
                y[4*p + 3] = w3p*(amc + jbmd);
            }
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q];
                const complex_t<T> b = y[q + 4];
                const complex_t<T> c = y[q + 8];
                const complex_t<T> d = y[q + 12];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q] = (apc + bpd) / 16;
                x[q + 4] = (amc - jbmd) / 16;
                x[q + 8] = (apc - bpd) / 16;
                x[q + 12] = (amc + jbmd) / 16;
            }
        } break;
    case 32 : {
            complex_t<T> y[32];
            for (int p = 0; p < 8; p++) {
                const complex_t<T> w1p = complex_t<T>(cos(p*theta0), -sin(p*theta0));
                const complex_t<T> w2p = w1p*w1p;
                const complex_t<T> w3p = w1p*w2p;
                const complex_t<T> a = x[p];
                const complex_t<T> b = x[p + 8];
                const complex_t<T> c = x[p + 16];
                const complex_t<T> d = x[p + 24];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[4*p + 0] =      apc +  bpd;
                y[4*p + 1] = w1p*(amc - jbmd);
                y[4*p + 2] = w2p*(apc -  bpd);
                y[4*p + 3] = w3p*(amc + jbmd);
            }

            // p = 0
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q];
                const complex_t<T> b = y[q + 8];
                const complex_t<T> c = y[q + 16];
                const complex_t<T> d = y[q + 24];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q] =  apc + bpd;
                x[q + 4] = amc - jbmd;
                x[q + 8] = apc - bpd;
                x[q + 12] = amc + jbmd;
            }
            // p = 1
            const T theta1 = 2*M_PI/8;
            const complex_t<T> w1p_e = complex_t<T>(cos(theta1), -sin(theta1));
            const complex_t<T> w2p_e = w1p_e*w1p_e;
            const complex_t<T> w3p_e = w1p_e*w2p_e;
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q + 4];
                const complex_t<T> b = y[q + 12];
                const complex_t<T> c = y[q + 20];
                const complex_t<T> d = y[q + 28];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q + 16] =  apc +  bpd;
                x[q + 20] = w1p_e*(amc - jbmd);
                x[q + 24] = w2p_e*(apc -  bpd);
                x[q + 28] = w3p_e*(amc + jbmd);
            }

            for (int q = 0; q < 16; q++) {
                const complex_t<T> a = x[q];
                const complex_t<T> b = x[q + 16];
                x[q + 0]  = (a + b)/32;
                x[q + 16] = (a - b)/32;
            }
        } break;
    default : {
            // The other situation
            fft_stockham_r4(N, x);
        } break;
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_ifft(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);
    const T theta0 = 2*M_PI/N;

    switch (N)
    {
    case 1 : {
        } return;
    case 2 : {
            // N = 2 is treated as the special situation
            const complex_t<T> a = x[0];
            const complex_t<T> b = x[1];
            x[0] = a + b;
            x[1] = a - b;
        } break;
    case 4 : {
            const complex_t<T> a = x[0];
            const complex_t<T> b = x[1];
            const complex_t<T> c = x[2];
            const complex_t<T> d = x[3];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[0] = apc +  bpd;
            x[1] = amc + jbmd;
            x[2] = apc -  bpd;
            x[3] = amc - jbmd;
        } break;
    case 8 : {
            // Pay attention, sin(theta0)
            const complex_t<T> w1p = complex_t<T>(cos(theta0), sin(theta0));
            const complex_t<T> w2p = w1p*w1p;
            const complex_t<T> w3p = w1p*w2p;
            complex_t<T> y[8];
            complex_t<T> a = x[0]; complex_t<T> b = x[2]; complex_t<T> c = x[4]; complex_t<T> d = x[6];
            complex_t<T>  apc =    a + c; complex_t<T>  amc =    a - c;
            complex_t<T>  bpd =    b + d; complex_t<T> jbmd = j*(b - d);
            y[0] = apc +  bpd; y[1] = amc + jbmd;
            y[2] = apc -  bpd; y[3] = amc - jbmd;
            a = x[1]; b = x[3]; c = x[5]; d = x[7];
            apc = a + c; amc = a - c;
            bpd = b + d; jbmd = j*(b - d);
            y[4] = apc +  bpd; y[5] = w1p * (amc + jbmd);
            y[6] = w2p * (apc -  bpd); y[7] = w3p * (amc - jbmd);
            a = y[0]; b = y[4];
            x[0] = a + b; x[4] = a - b;
            a = y[1]; b = y[5];
            x[1] = a + b; x[5] = a - b;
            a = y[2]; b = y[6];
            x[2] = a + b; x[6] = a - b;
            a = y[3]; b = y[7];
            x[3] = a + b; x[7] = a - b;
        } break;
    case 16 : {
            complex_t<T> y[16];
            for (int p = 0; p < 4; p++) {
                // Pay attention, sin(theta0)
                const complex_t<T> w1p = complex_t<T>(cos(p*theta0), sin(p*theta0));
                const complex_t<T> w2p = w1p*w1p;
                const complex_t<T> w3p = w1p*w2p;

                const complex_t<T> a = x[p];
                const complex_t<T> b = x[p + 4];
                const complex_t<T> c = x[p + 8];
                const complex_t<T> d = x[p + 12];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[4*p] = apc + bpd;
                y[4*p + 1] = w1p*(amc + jbmd);
                y[4*p + 2] = w2p*(apc - bpd);
                y[4*p + 3] = w3p*(amc - jbmd);
            }
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q];
                const complex_t<T> b = y[q + 4];
                const complex_t<T> c = y[q + 8];
                const complex_t<T> d = y[q + 12];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q] = apc + bpd;
                x[q + 4] = amc + jbmd;
                x[q + 8] = apc - bpd;
                x[q + 12] = amc - jbmd;
            }
        } break;
    case 32 : {
            complex_t<T> y[32];
            for (int p = 0; p < 8; p++) {
                // Pay attention, sin(theta0
                const complex_t<T> w1p = complex_t<T>(cos(p*theta0), sin(p*theta0));
                const complex_t<T> w2p = w1p*w1p;
                const complex_t<T> w3p = w1p*w2p;
                const complex_t<T> a = x[p];
                const complex_t<T> b = x[p + 8];
                const complex_t<T> c = x[p + 16];
                const complex_t<T> d = x[p + 24];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[4*p + 0] =      apc +  bpd;
                y[4*p + 1] = w1p*(amc + jbmd);
                y[4*p + 2] = w2p*(apc -  bpd);
                y[4*p + 3] = w3p*(amc - jbmd);
            }

            // p = 0
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q];
                const complex_t<T> b = y[q + 8];
                const complex_t<T> c = y[q + 16];
                const complex_t<T> d = y[q + 24];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q] =  apc + bpd;
                x[q + 4] = amc + jbmd;
                x[q + 8] = apc - bpd;
                x[q + 12] = amc - jbmd;
            }
            // p = 1
            const T theta1 = 2*M_PI/8;
            // Pay attention, sin(theta0
            const complex_t<T> w1p_e = complex_t<T>(cos(theta1), sin(theta1));
            const complex_t<T> w2p_e = w1p_e*w1p_e;
            const complex_t<T> w3p_e = w1p_e*w2p_e;
            for (int q = 0; q < 4; q++) {
                const complex_t<T> a = y[q + 4];
                const complex_t<T> b = y[q + 12];
                const complex_t<T> c = y[q + 20];
                const complex_t<T> d = y[q + 28];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                x[q + 16] =  apc +  bpd;
                x[q + 20] = w1p_e*(amc + jbmd);
                x[q + 24] = w2p_e*(apc -  bpd);
                x[q + 28] = w3p_e*(amc - jbmd);
            }

            for (int q = 0; q < 16; q++) {
                const complex_t<T> a = x[q];
                const complex_t<T> b = x[q + 16];
                x[q + 0]  = a + b;
                x[q + 16] = a - b;
            }
        } break;
    default : {
            // The other situation
            ifft_stockham_r4(N, x);
        } break;
    }
}

#endif
