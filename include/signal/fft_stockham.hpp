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

#ifndef _FFT_STOCKHAM_HPP_
#define _FFT_STOCKHAM_HPP_          1

#include <cmath>
#include <utility>
#include "common/complex_t.h"

template<typename T>
void fft1_stockham(int n, int s, int q, complex_t<T> *x, complex_t<T> *y);

// n : sequence length
// s : stride
// q : selection of even or odd
// x : input/output sequence
// y : work area
template<typename T>
void fft0_stockham(int n, int s, int q, complex_t<T> *x, complex_t<T> *y)
{
    const int m = n/2;
    const T theta0 = 2*std::M_PI/n;

    if (n == 1) {}
    else {
        for (int p = 0; p < m; p++) {
            // Butterfly operation and composition of even components and odd components
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            const complex_t<T> a = x[q + s*(p + 0)];
            const complex_t<T> b = x[q + s*(p + m)];
            y[q + s*(2*p + 0)] =  a + b;
            y[q + s*(2*p + 1)] = (a - b) * wp;
        }
        fft1_stockham<T>(n/2, 2*s, q + 0, y, x); // Even place FFT (y:input, x:output)
        fft1_stockham<T>(n/2, 2*s, q + s, y, x); // Odd place FFT (y:input, x:output)
    }
}

// n : sequence length
// s : stride
// q : selection of even or odd
// x : input/output sequence
// y : work area
template<typename T>
void fft1_stockham(int n, int s, int q, complex_t<T> *x, complex_t<T> *y)
{
    const int m = n/2;
    const T theta0 = 2*std::M_PI/n;

    if (n == 1) { y[q] = x[q]; }
    else {
        for (int p = 0; p < m; p++) {
            // Butterfly Operation and composition of even components and odd components
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            const complex_t<T> a = x[q + s*(p + 0)];
            const complex_t<T> b = x[q + s*(p + m)];
            y[q + s*(2*p + 0)] =  a + b;
            y[q + s*(2*p + 1)] = (a - b) * wp;
        }
        fft0_stockham<T>(n/2, 2*s, q + 0, y, x); // Even place FFT (y:input/output, x:work area)
        fft0_stockham<T>(n/2, 2*s, q + s, y, x); // Odd place FFT (y:input/output, x:work area)
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void fft_stockham(int N, complex_t<T> *x)
{
    complex_t* y = new complex_t[N];
    fft0_stockham<T>(N, 1, 0, x, y);
    delete []y;
    for (int k = 0; k < n; k++) {
        x[k] /= N;
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void ifft_stockham(int N, complex_t<T> *x)
{
    for (int k = 0; k < N; k++) {
        x[k].conj();
    }

    complex_t *y = new complex_t[N];
    fft0_stockham<T>(N, 1, 0, x, y);
    delete []y;

    for (int p = 0; p < N; p++) {
        x[p] = conj(x[p]);
    }
}

#endif
