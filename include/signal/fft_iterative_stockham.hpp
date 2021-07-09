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

#ifndef _FFT_ITERATIVE_STOCKHAM_HPP_
#define _FFT_ITERATIVE_STOCKHAM_HPP_          1

#include <cmath>
#include <utility>
#include "common/complex_t.h"

// n : sequence length
// s : stride
// eo : x is output if eo == 0, y is output if eo == 1
// x : input sequence(or output sequence if eo == 0)
// y : work area(or output sequence if eo == 1)
template<typename T>
void fft0_stockham_iterative(int n, int s, bool eo, complex_t<T> *x, complex_t<T> *y)
{
    const int m = n/2;
    const T theta0 = 2*M_PI/n;

#if 0
    if (n == 1) {
        if (eo) {
            for (int q = 0; q < s; q++) {
                y[q] = x[q];
            }
        }
    } else {
        for (int p = 0; p < m; p++) {
            // Butterfly operation and composition of even components and odd components
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            for (int q = 0; q < s; q++) {
                const complex_t a = x[q + s*(p + 0)];
                const complex_t b = x[q + s*(p + m)];
                y[q + s*(2*p + 0)] =  a + b;
                y[q + s*(2*p + 1)] = (a - b) * wp;
            }
        }

        fft0_stockham_iterative<T>(n/2, 2*s, !eo, y, x);
    }
#endif
    // Substitute a calculation result into the destination array beforehand, so remove this copy
    // Because n == 1 in fft0 does not calculate anything thus could reduce the accesses to the array
    if (n == 2) {
        complex_t<T> *z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            const complex_t<T> a = x[q+0];
            const complex_t<T> b = x[q+s];
            z[q + 0] = a + b;
            z[q + s] = a - b;
        }
    } else if (n >= 4) {
        for (int p = 0; p < m; p++) {
            // Butterfly operation and composition of even components and odd components
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            for (int q = 0; q < s; q++) {
                const complex_t<T> a = x[q + s*(p + 0)];
                const complex_t<T> b = x[q + s*(p + m)];
                y[q + s*(2*p + 0)] =  a + b;
                y[q + s*(2*p + 1)] = (a - b) * wp;
            }
        }

        fft0_stockham_iterative<T>(n/2, 2*s, !eo, y, x);
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void fft_stockham_iterative(int N, complex_t<T> *x)
{
    complex_t<T>* y = new complex_t<T>[N];
    fft0_stockham_iterative<T>(N, 1, 0, x, y);
    delete []y;
    for (int k = 0; k < N; k++) {
        x[k] /= N;
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void ifft_stockham_iterative(int N, complex_t<T> *x)
{
    for (int k = 0; k < N; k++) {
        x[k].conj();
    }

    complex_t<T> *y = new complex_t<T>[N];
    fft0_stockham_iterative<T>(N, 1, 0, x, y);
    delete []y;

    for (int p = 0; p < N; p++) {
        x[p] = conj(x[p]);
    }
}

#endif
