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

#ifndef _MY_FFT_AVX_HPP_
#define _MY_FFT_AVX_HPP_            1

#include <cmath>
#include <utility>
#include "common/complex_t.h"

/////////////////////////////////////////////////
//    my_fft.hpp AVX or AVX512 optimization    //
/////////////////////////////////////////////////

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_fft_avx(int N, complex_t<T> *x)
{
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_ifft_avx(int N, complex_t<T> *x)
{
}

#endif
