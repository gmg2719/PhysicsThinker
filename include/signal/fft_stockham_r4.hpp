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

#ifndef _FFT_STOCKHAM_R4_H_
#define _FFT_STOCKHAM_R4_H_         1

#include <cmath>
#include <utility>
#include "common/complex_t.h"

// n : sequence length
// s : stride
// eo : x is output if eo == 0, y is output if eo == 1
// x : input sequence(or output sequence if eo == 0)
// y : work area(or output sequence if eo == 1)
template<typename T>
void fft0_stockham_r4(int n, int s, bool eo, complex_t<T> *x, complex_t<T> *y)
{
    static const complex_t<T> j = complex_t<T>(0, 1);
    const int n0 = 0;
    const int n1 = n/4;
    const int n2 = n/2;
    const int n3 = n1 + n2;
    const T theta0 = 2*M_PI/n;

    if (n == 1) { if (eo) for (int q = 0; q < s; q++) y[q] = x[q]; }
    else if (n == 2) {
        for (int q = 0; q < s; q++) {
            const complex_t<T> a = x[q + 0];
            const complex_t<T> b = x[q + s];
            y[q + 0] = a + b;
            y[q + s] = a - b;
        }
        fft0_stockham_r4(1, 2*s, !eo, y, x);
    }
    else if (n > 2) {
        for (int p = 0; p < n1; p++) {
            const complex_t<T> w1p = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            const complex_t<T> w2p = w1p*w1p;
            const complex_t<T> w3p = w1p*w2p;
            for (int q = 0; q < s; q++) {
                const complex_t<T> a = x[q + s*(p + n0)];
                const complex_t<T> b = x[q + s*(p + n1)];
                const complex_t<T> c = x[q + s*(p + n2)];
                const complex_t<T> d = x[q + s*(p + n3)];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[q + s*(4*p + 0)] =      apc +  bpd;
                y[q + s*(4*p + 1)] = w1p*(amc - jbmd);
                y[q + s*(4*p + 2)] = w2p*(apc -  bpd);
                y[q + s*(4*p + 3)] = w3p*(amc + jbmd);
            }
        }
        fft0_stockham_r4(n/4, 4*s, !eo, y, x);
    }
}

template<typename T>
void ifft0_stockham_r4(int n, int s, bool eo, complex_t<T> *x, complex_t<T> *y)
{
    static const complex_t<T> j = complex_t<T>(0, 1);
    const int n0 = 0;
    const int n1 = n/4;
    const int n2 = n/2;
    const int n3 = n1 + n2;
    const T theta0 = 2*M_PI/n;

    if (n == 1) { if (eo) for (int q = 0; q < s; q++) y[q] = x[q]; }
    else if (n == 2) {
        for (int q = 0; q < s; q++) {
            const complex_t<T> a = x[q + 0];
            const complex_t<T> b = x[q + s];
            y[q + 0] = a + b;
            y[q + s] = a - b;
        }
        ifft0_stockham_r4(1, 2*s, !eo, y, x);
    }
    else if (n > 2) {
        for (int p = 0; p < n1; p++) {
            const complex_t<T> w1p = complex_t<T>(cos(p*theta0), sin(p*theta0));
            const complex_t<T> w2p = w1p*w1p;
            const complex_t<T> w3p = w1p*w2p;
            for (int q = 0; q < s; q++) {
                const complex_t<T> a = x[q + s*(p + n0)];
                const complex_t<T> b = x[q + s*(p + n1)];
                const complex_t<T> c = x[q + s*(p + n2)];
                const complex_t<T> d = x[q + s*(p + n3)];
                const complex_t<T>  apc =    a + c;
                const complex_t<T>  amc =    a - c;
                const complex_t<T>  bpd =    b + d;
                const complex_t<T> jbmd = j*(b - d);
                y[q + s*(4*p + 0)] =      apc +  bpd;
                y[q + s*(4*p + 1)] = w1p*(amc + jbmd);
                y[q + s*(4*p + 2)] = w2p*(apc -  bpd);
                y[q + s*(4*p + 3)] = w3p*(amc - jbmd);
            }
        }
        ifft0_stockham_r4(n/4, 4*s, !eo, y, x);
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void fft_stockham_r4(int N, complex_t<T> *x)
{
    complex_t<T> *y = new complex_t<T>[N];
    fft0_stockham_r4<T>(N, 1, 0, x, y);
    delete []y;
    for (int k = 0; k < N; k++) {
        x[k] /= N;
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void ifft_stockham_r4(int N, complex_t<T> *x)
{
    complex_t<T> *y = new complex_t<T>[N];
    ifft0_stockham_r4<T>(N, 1, 0, x, y);
    delete []y;
}

#endif
