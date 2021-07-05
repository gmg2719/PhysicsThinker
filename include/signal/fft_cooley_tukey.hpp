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

#ifndef _FFT_COOLEY_TUKEY_HPP_
#define _FFT_COOLEY_TUKEY_HPP_         1

#include <cmath>
#include <utility>
#include "common/complex_t.h"

// bit reversal sorting
// N : sequence length
// x : input/output sequence
template <typename T>
void bit_reverse(int N, complex_t<T> *x)
{
    for (int i = 0, j = 1; j < N - 1; j++) {
        for (int k = N >> 1; k > (i ^= k); k >>= 1) {
            (void)0;
        }
        if (i < j)  std::swap(x[i], x[j]);
    }
}

// N : sequence length
// q : block start point (initial offset value is 0)
// x : input/output sequence
template<typename T>
void fft_original(int N, int q, complex_t<T> *x)
{
    const int m = N/2;
    const T theta0 = 2 * std::M_PI / N;

    if (N > 1) {
        for (int p = 0; p < m; p++) {
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            const complex_t<T> a = x[q+p];
            const complex_t<T> b = x[q+p+m];

            x[q+p] = a + b;
            x[q+p+m] = (a-b)*wp;
        }

        F(N/2, q, x);       // Even position FFT
        F(N/2, q+m, x);     // Odd position FFT
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void fft(int N, complex_t<T> *x)
{
    fft_original<T>(N, 0, x);
    bit_reverse<T>(N, x);
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void ifft(int N, complex_t<T> *x)
{
    for (int k = 0; k < N; k++) {
        x[k].conj();
    }
    fft_original<T>(N, 0, x);
    bit_reverse<T>(N, x);
    for (int p = 0; p < N; p++) {
        x[p] = conj(x[p]) / T(N);
    }
}

// N : sequence length
// s : stride
// q : block start point (initial offset value is 0)
// d : bit reversal point of q
// x : input/output sequence
template<typename T>
void fft_original_reversal(int N, int s, int q, int d, complex_t<T> *x)
{
    const int m = N/2;
    const T theta0 = 2 * std::M_PI / N;

    if (N > 1) {
        for (int p = 0; p < m; p++) {
            const complex_t<T> wp = complex_t<T>(cos(p*theta0), -sin(p*theta0));
            const complex_t<T> a = x[q+p];
            const complex_t<T> b = x[q+p+m];

            x[q+p] = a + b;
            x[q+p+m] = (a-b)*wp;
        }

        F(N/2, 2*s, q, d, x);       // Even position FFT
        F(N/2, 2*s, q+m, d+s, x);     // Odd position FFT
    } else if (q > d) {
        // Bit reversal sorting
        std::swap(x[q], x[d]);
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void fft_n(int N, complex_t<T> *x)
{
    fft_original_reversal<T>(N, 1, 0, 0, x);
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void ifft_n(int N, complex_t<T> *x)
{
    for (int k = 0; k < N; k++) {
        x[k].conj();
    }

    fft_original_reversal<T>(N, 1, 0, 0, x);

    for (int p = 0; p < N; p++) {
        x[p] = conj(x[p]) / T(N);
    }
}

#endif
