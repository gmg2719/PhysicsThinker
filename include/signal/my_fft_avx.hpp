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
#include "my_simd.h"

#define SQRT2_CONSTANT          1.41421356237309504880  // sqrt(2)
#define SQRT1_2_CONSTANT        0.70710678118654752440  // 1/sqrt(2)

/////////////////////////////////////////////////
//      some useful SIMD functions for FFT     //
/////////////////////////////////////////////////

static inline __m128 v8xpz_f(const __m128 xy)
{
    const __m128 rr = {1.0, 1.0, SQRT1_2_CONSTANT, SQRT1_2_CONSTANT};
    const __m128 zm = {0.0, 0.0, 0.0, -0.0};
    __m128 xmy_tmp = _mm_xor_ps(zm, xy);
    __m128 xmy = _mm_shuffle_ps(_mm_setzero_ps(), xmy_tmp, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_mul_ps(rr, _mm_add_ps(xy, xmy));
}

static inline __m128 w8xpz_f(const __m128 xy)
{
    const __m128 rr = {1.0, 1.0, SQRT1_2_CONSTANT, SQRT1_2_CONSTANT};
    const __m128 zm = {0.0, 0.0, 0.0, -0.0};
    __m128 xmy_tmp = _mm_shuffle_ps(_mm_setzero_ps(), xy, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 xmy = _mm_xor_ps(zm, xmy_tmp);
    return _mm_mul_ps(rr, _mm_add_ps(xy, xmy));
}

static inline __m128 jxpz(const __m128 xy)
{
    static const __m128 zm = {0.0, -0.0, 0.0, -0.0};
    __m128 xmy = _mm_xor_ps(zm, xy);
    return _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 0, 1));
}

static inline __m256 duppz2_lo(const __m128 x)
{
    __m128 y = _mm_movelh_ps(x, x);
    return _mm256_broadcast_ps(&y);
}

static inline __m256 mulpz2(const __m256 ab, const __m256 xy)
{
    __m256 aa = _mm256_moveldup_ps(ab);
    __m256 bb = _mm256_movehdup_ps(ab);
    __m256 yx = _mm256_shuffle_ps(xy, xy, 0xb1);
#ifdef __FMA__
    return _mm256_fmaddsub_ps(aa, xy, _mm256_mul_ps(bb, yx));
#else
    return _mm256_addsub_ps(_mm256_mul_ps(aa, xy), _mm256_mul_ps(bb, yx));
#endif
}

static inline __m256 jxpz2(const __m256 xy)
{
    static const __m256 zm = {0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};
    __m256 xmy = _mm256_xor_ps(zm, xy);
    return _mm256_shuffle_ps(xmy, xmy, 0xb1); 
}

static inline __m128 mulpz_lh(const __m128 a, const __m128 b)
{
    __m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0));
    __m128 bb = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));
    __m128 yx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef __FMA__
    return _mm_fmaddsub_ps(aa, b, _mm_mul_ps(bb, yx));
#else
    return _mm_addsub_ps(_mm_mul_ps(aa, b), _mm_mul_ps(bb, yx));
#endif
}

/////////////////////////////////////////////////
//                                             //
//    my_fft.hpp AVX or AVX512 optimization    //
//       Forward : do not multiply the N       //
//                                             //
/////////////////////////////////////////////////

template<typename T>
struct my_fft_avx_whole
{
    complex_t<T> w1p8_fwd;
    complex_t<T> w2p8_fwd;
    complex_t<T> w3p8_fwd;
    __m128 w1p8_fwd_nse[2];
    __m128 w2p8_fwd_nse[2];
    __m128 w3p8_fwd_nse[2];
    complex_t<T> w1p16_fwd[4];
    complex_t<T> w2p16_fwd[4];
    complex_t<T> w3p16_fwd[4];
    __m128 w1p16_fwd_avx[2];
    __m128 w2p16_fwd_avx[2];
    __m128 w3p16_fwd_avx[2];
    __m128 w1p16_fwd_nse[4];
    __m128 w2p16_fwd_nse[4];
    __m128 w3p16_fwd_nse[4];
    complex_t<T> w1p32_fwd[8];
    complex_t<T> w2p32_fwd[8];
    complex_t<T> w3p32_fwd[8];
    __m128 w1p32_fwd_avx[4];
    __m128 w2p32_fwd_avx[4];
    __m128 w3p32_fwd_avx[4];
    __m128 w1p32_fwd_nse[8];
    __m128 w2p32_fwd_nse[8];
    __m128 w3p32_fwd_nse[8];
    complex_t<T> w1p64_fwd[16];
    complex_t<T> w2p64_fwd[16];
    complex_t<T> w3p64_fwd[16];
    __m128 w1p64_fwd_avx[8];
    __m128 w2p64_fwd_avx[8];
    __m128 w3p64_fwd_avx[8];
    __m128 w1p64_fwd_nse[16];
    __m128 w2p64_fwd_nse[16];
    __m128 w3p64_fwd_nse[16];
    complex_t<T> w1p128_fwd[32];
    complex_t<T> w2p128_fwd[32];
    complex_t<T> w3p128_fwd[32];
    __m128 w1p128_fwd_avx[16];
    __m128 w2p128_fwd_avx[16];
    __m128 w3p128_fwd_avx[16];
    __m128 w1p128_fwd_nse[32];
    __m128 w2p128_fwd_nse[32];
    __m128 w3p128_fwd_nse[32];
    complex_t<T> w1p256_fwd[64];
    complex_t<T> w2p256_fwd[64];
    complex_t<T> w3p256_fwd[64];
    __m128 w1p256_fwd_avx[32];
    __m128 w2p256_fwd_avx[32];
    __m128 w3p256_fwd_avx[32];
    __m128 w1p256_fwd_nse[64];
    __m128 w2p256_fwd_nse[64];
    __m128 w3p256_fwd_nse[64];
    complex_t<T> w1p512_fwd[128];
    complex_t<T> w2p512_fwd[128];
    complex_t<T> w3p512_fwd[128];
    __m128 w1p512_fwd_avx[64];
    __m128 w2p512_fwd_avx[64];
    __m128 w3p512_fwd_avx[64];
    __m128 w1p512_fwd_nse[128];
    __m128 w2p512_fwd_nse[128];
    __m128 w3p512_fwd_nse[128];
    complex_t<T> w1p1024_fwd[256];
    complex_t<T> w2p1024_fwd[256];
    complex_t<T> w3p1024_fwd[256];
    __m128 w1p1024_fwd_avx[128];
    __m128 w2p1024_fwd_avx[128];
    __m128 w3p1024_fwd_avx[128];
    __m128 w1p1024_fwd_nse[256];
    __m128 w2p1024_fwd_nse[256];
    __m128 w3p1024_fwd_nse[256];
    complex_t<T> w1p2048_fwd[512];
    complex_t<T> w2p2048_fwd[512];
    complex_t<T> w3p2048_fwd[512];
    __m128 w1p2048_fwd_avx[256];
    __m128 w2p2048_fwd_avx[256];
    __m128 w3p2048_fwd_avx[256];
    __m128 w1p2048_fwd_nse[512];
    __m128 w2p2048_fwd_nse[512];
    __m128 w3p2048_fwd_nse[512];
    complex_t<T> w1p4096_fwd[1024];
    complex_t<T> w2p4096_fwd[1024];
    complex_t<T> w3p4096_fwd[1024];
    __m128 w1p4096_fwd_avx[512];
    __m128 w2p4096_fwd_avx[512];
    __m128 w3p4096_fwd_avx[512];
    // The construct interface
    my_fft_avx_whole(int N);
    // FFT forward operations
    inline void my_fft_8points(int N, complex_t<T> *x);
    inline void my_fft_16points(int N, complex_t<T> *x);
    inline void my_fft_32points(int N, complex_t<T> *x);
    inline void my_fft_64points(int N, complex_t<T> *x);
    inline void my_fft_128points(int N, complex_t<T> *x);
    inline void my_fft_256points(int N, complex_t<T> *x);
    inline void my_fft_512points(int N, complex_t<T> *x);
    inline void my_fft_1024points(int N, complex_t<T> *x);
    inline void my_fft_2048points(int N, complex_t<T> *x);
    inline void my_fft_4096points(int N, complex_t<T> *x);
    // FFT backward operations
    inline void my_ifft_8points(int N, complex_t<T> *x);
    inline void my_ifft_16points(int N, complex_t<T> *x);
    inline void my_ifft_32points(int N, complex_t<T> *x);
    inline void my_ifft_64points(int N, complex_t<T> *x);
    inline void my_ifft_128points(int N, complex_t<T> *x);
    inline void my_ifft_256points(int N, complex_t<T> *x);
    inline void my_ifft_512points(int N, complex_t<T> *x);
    inline void my_ifft_1024points(int N, complex_t<T> *x);
    inline void my_ifft_2048points(int N, complex_t<T> *x);
    inline void my_ifft_4096points(int N, complex_t<T> *x);
    void my_fft(int N, complex_t<T> *x);
    void my_ifft(int N, complex_t<T> *x);
};

template<typename T>
my_fft_avx_whole<T>::my_fft_avx_whole(int N)
{
    const T theta = 2*M_PI/8;
    w1p8_fwd = complex_t<T>(cos(theta), -sin(theta));
    w2p8_fwd = w1p8_fwd * w1p8_fwd;
    w3p8_fwd = w1p8_fwd * w2p8_fwd;
    w1p8_fwd_nse[0] = _mm_set_ps(-0.0, 1.0, -0.0, 1.0);
    w2p8_fwd_nse[0] = _mm_set_ps(-0.0, 1.0, -0.0, 1.0);
    w3p8_fwd_nse[0] = _mm_set_ps(-0.0, 1.0, -0.0, 1.0);
    w1p8_fwd_nse[1] = _mm_set_ps(w1p8_fwd.Im, w1p8_fwd.Re, w1p8_fwd.Im, w1p8_fwd.Re);
    w2p8_fwd_nse[1] = _mm_set_ps(w2p8_fwd.Im, w2p8_fwd.Re, w2p8_fwd.Im, w2p8_fwd.Re);
    w3p8_fwd_nse[1] = _mm_set_ps(w3p8_fwd.Im, w3p8_fwd.Re, w3p8_fwd.Im, w3p8_fwd.Re);

    const T theta0 = 2*M_PI/16;
    for (int p = 0; p < 4; p++) {
        w1p16_fwd[p] = complex_t<T>(cos(p*theta0), -sin(p*theta0));
        w2p16_fwd[p] = w1p16_fwd[p] * w1p16_fwd[p];
        w3p16_fwd[p] = w1p16_fwd[p] * w2p16_fwd[p];
        w1p16_fwd_nse[p] = _mm_set_ps(w1p16_fwd[p].Im, w1p16_fwd[p].Re, w1p16_fwd[p].Im, w1p16_fwd[p].Re);
        w2p16_fwd_nse[p] = _mm_set_ps(w2p16_fwd[p].Im, w2p16_fwd[p].Re, w2p16_fwd[p].Im, w2p16_fwd[p].Re);
        w3p16_fwd_nse[p] = _mm_set_ps(w3p16_fwd[p].Im, w3p16_fwd[p].Re, w3p16_fwd[p].Im, w3p16_fwd[p].Re);
    }
    for (int p = 0; p < 4; p+=2) {
        w1p16_fwd_avx[p/2] = _mm_loadu_ps(&(w1p16_fwd[p].Re));
        w2p16_fwd_avx[p/2] = _mm_loadu_ps(&(w2p16_fwd[p].Re));
        w3p16_fwd_avx[p/2] = _mm_loadu_ps(&(w3p16_fwd[p].Re));
    }

    const T theta1 = 2*M_PI/32;
    for (int p = 0; p < 8; p++) {
        w1p32_fwd[p] = complex_t<T>(cos(p*theta1), -sin(p*theta1));
        w2p32_fwd[p] = w1p32_fwd[p] * w1p32_fwd[p];
        w3p32_fwd[p] = w1p32_fwd[p] * w2p32_fwd[p];
        w1p32_fwd_nse[p] = _mm_set_ps(w1p32_fwd[p].Im, w1p32_fwd[p].Re, w1p32_fwd[p].Im, w1p32_fwd[p].Re);
        w2p32_fwd_nse[p] = _mm_set_ps(w2p32_fwd[p].Im, w2p32_fwd[p].Re, w2p32_fwd[p].Im, w2p32_fwd[p].Re);
        w3p32_fwd_nse[p] = _mm_set_ps(w3p32_fwd[p].Im, w3p32_fwd[p].Re, w3p32_fwd[p].Im, w3p32_fwd[p].Re);
    }
    for (int p = 0; p < 8; p+=2) {
        w1p32_fwd_avx[p/2] = _mm_loadu_ps(&(w1p32_fwd[p].Re));
        w2p32_fwd_avx[p/2] = _mm_loadu_ps(&(w2p32_fwd[p].Re));
        w3p32_fwd_avx[p/2] = _mm_loadu_ps(&(w3p32_fwd[p].Re));
    }

    const T theta2 = 2*M_PI/64;
    for (int p = 0; p < 16; p++) {
        w1p64_fwd[p] = complex_t<T>(cos(p*theta2), -sin(p*theta2));
        w2p64_fwd[p] = w1p64_fwd[p] * w1p64_fwd[p];
        w3p64_fwd[p] = w1p64_fwd[p] * w2p64_fwd[p];
        w1p64_fwd_nse[p] = _mm_set_ps(w1p64_fwd[p].Im, w1p64_fwd[p].Re, w1p64_fwd[p].Im, w1p64_fwd[p].Re);
        w2p64_fwd_nse[p] = _mm_set_ps(w2p64_fwd[p].Im, w2p64_fwd[p].Re, w2p64_fwd[p].Im, w2p64_fwd[p].Re);
        w3p64_fwd_nse[p] = _mm_set_ps(w3p64_fwd[p].Im, w3p64_fwd[p].Re, w3p64_fwd[p].Im, w3p64_fwd[p].Re);
    }
    for (int p = 0; p < 16; p+=2) {
        w1p64_fwd_avx[p/2] = _mm_loadu_ps(&(w1p64_fwd[p].Re));
        w2p64_fwd_avx[p/2] = _mm_loadu_ps(&(w2p64_fwd[p].Re));
        w3p64_fwd_avx[p/2] = _mm_loadu_ps(&(w3p64_fwd[p].Re));
    }

    const T theta3 = 2*M_PI/128;
    for (int p = 0; p < 32; p++) {
        w1p128_fwd[p] = complex_t<T>(cos(p*theta3), -sin(p*theta3));
        w2p128_fwd[p] = w1p128_fwd[p] * w1p128_fwd[p];
        w3p128_fwd[p] = w1p128_fwd[p] * w2p128_fwd[p];
        w1p128_fwd_nse[p] = _mm_set_ps(w1p128_fwd[p].Im, w1p128_fwd[p].Re, w1p128_fwd[p].Im, w1p128_fwd[p].Re);
        w2p128_fwd_nse[p] = _mm_set_ps(w2p128_fwd[p].Im, w2p128_fwd[p].Re, w2p128_fwd[p].Im, w2p128_fwd[p].Re);
        w3p128_fwd_nse[p] = _mm_set_ps(w3p128_fwd[p].Im, w3p128_fwd[p].Re, w3p128_fwd[p].Im, w3p128_fwd[p].Re);
    }
    for (int p = 0; p < 32; p+=2) {
        w1p128_fwd_avx[p/2] = _mm_loadu_ps(&(w1p128_fwd[p].Re));
        w2p128_fwd_avx[p/2] = _mm_loadu_ps(&(w2p128_fwd[p].Re));
        w3p128_fwd_avx[p/2] = _mm_loadu_ps(&(w3p128_fwd[p].Re));
    }

    const T theta4 = 2*M_PI/256;
    for (int p = 0; p < 64; p++) {
        w1p256_fwd[p] = complex_t<T>(cos(p*theta4), -sin(p*theta4));
        w2p256_fwd[p] = w1p256_fwd[p] * w1p256_fwd[p];
        w3p256_fwd[p] = w1p256_fwd[p] * w2p256_fwd[p];
        w1p256_fwd_nse[p] = _mm_set_ps(w1p256_fwd[p].Im, w1p256_fwd[p].Re, w1p256_fwd[p].Im, w1p256_fwd[p].Re);
        w2p256_fwd_nse[p] = _mm_set_ps(w2p256_fwd[p].Im, w2p256_fwd[p].Re, w2p256_fwd[p].Im, w2p256_fwd[p].Re);
        w3p256_fwd_nse[p] = _mm_set_ps(w3p256_fwd[p].Im, w3p256_fwd[p].Re, w3p256_fwd[p].Im, w3p256_fwd[p].Re);
    }
    for (int p = 0; p < 64; p+=2) {
        w1p256_fwd_avx[p/2] = _mm_loadu_ps(&(w1p256_fwd[p].Re));
        w2p256_fwd_avx[p/2] = _mm_loadu_ps(&(w2p256_fwd[p].Re));
        w3p256_fwd_avx[p/2] = _mm_loadu_ps(&(w3p256_fwd[p].Re));
    }

    const T theta5 = 2*M_PI/512;
    for (int p = 0; p < 128; p++) {
        w1p512_fwd[p] = complex_t<T>(cos(p*theta5), -sin(p*theta5));
        w2p512_fwd[p] = w1p512_fwd[p] * w1p512_fwd[p];
        w3p512_fwd[p] = w1p512_fwd[p] * w2p512_fwd[p];
        w1p512_fwd_nse[p] = _mm_set_ps(w1p512_fwd[p].Im, w1p512_fwd[p].Re, w1p512_fwd[p].Im, w1p512_fwd[p].Re);
        w2p512_fwd_nse[p] = _mm_set_ps(w2p512_fwd[p].Im, w2p512_fwd[p].Re, w2p512_fwd[p].Im, w2p512_fwd[p].Re);
        w3p512_fwd_nse[p] = _mm_set_ps(w3p512_fwd[p].Im, w3p512_fwd[p].Re, w3p512_fwd[p].Im, w3p512_fwd[p].Re);
    }
    for (int p = 0; p < 128; p+=2) {
        w1p512_fwd_avx[p/2] = _mm_loadu_ps(&(w1p512_fwd[p].Re));
        w2p512_fwd_avx[p/2] = _mm_loadu_ps(&(w2p512_fwd[p].Re));
        w3p512_fwd_avx[p/2] = _mm_loadu_ps(&(w3p512_fwd[p].Re));
    }

    const T theta6 = 2*M_PI/1024;
    for (int p = 0; p < 256; p++) {
        w1p1024_fwd[p] = complex_t<T>(cos(p*theta6), -sin(p*theta6));
        w2p1024_fwd[p] = w1p1024_fwd[p] * w1p1024_fwd[p];
        w3p1024_fwd[p] = w1p1024_fwd[p] * w2p1024_fwd[p];
        w1p1024_fwd_nse[p] = _mm_set_ps(w1p1024_fwd[p].Im, w1p1024_fwd[p].Re, w1p1024_fwd[p].Im, w1p1024_fwd[p].Re);
        w2p1024_fwd_nse[p] = _mm_set_ps(w2p1024_fwd[p].Im, w2p1024_fwd[p].Re, w2p1024_fwd[p].Im, w2p1024_fwd[p].Re);
        w3p1024_fwd_nse[p] = _mm_set_ps(w3p1024_fwd[p].Im, w3p1024_fwd[p].Re, w3p1024_fwd[p].Im, w3p1024_fwd[p].Re);
    }
    for (int p = 0; p < 256; p+=2) {
        w1p1024_fwd_avx[p/2] = _mm_loadu_ps(&(w1p1024_fwd[p].Re));
        w2p1024_fwd_avx[p/2] = _mm_loadu_ps(&(w2p1024_fwd[p].Re));
        w3p1024_fwd_avx[p/2] = _mm_loadu_ps(&(w3p1024_fwd[p].Re));
    }

    const T theta7 = 2*M_PI/2048;
    for (int p = 0; p < 512; p++) {
        w1p2048_fwd[p] = complex_t<T>(cos(p*theta7), -sin(p*theta7));
        w2p2048_fwd[p] = w1p2048_fwd[p] * w1p2048_fwd[p];
        w3p2048_fwd[p] = w1p2048_fwd[p] * w2p2048_fwd[p];
        w1p2048_fwd_nse[p] = _mm_set_ps(w1p2048_fwd[p].Im, w1p2048_fwd[p].Re, w1p2048_fwd[p].Im, w1p2048_fwd[p].Re);
        w2p2048_fwd_nse[p] = _mm_set_ps(w2p2048_fwd[p].Im, w2p2048_fwd[p].Re, w2p2048_fwd[p].Im, w2p2048_fwd[p].Re);
        w3p2048_fwd_nse[p] = _mm_set_ps(w3p2048_fwd[p].Im, w3p2048_fwd[p].Re, w3p2048_fwd[p].Im, w3p2048_fwd[p].Re);
    }
    for (int p = 0; p < 512; p+=2) {
        w1p2048_fwd_avx[p/2] = _mm_loadu_ps(&(w1p2048_fwd[p].Re));
        w2p2048_fwd_avx[p/2] = _mm_loadu_ps(&(w2p2048_fwd[p].Re));
        w3p2048_fwd_avx[p/2] = _mm_loadu_ps(&(w3p2048_fwd[p].Re));
    }

    const T theta8 = 2*M_PI/4096;
    for (int p = 0; p < 1024; p++) {
        w1p4096_fwd[p] = complex_t<T>(cos(p*theta8), -sin(p*theta8));
        w2p4096_fwd[p] = w1p4096_fwd[p] * w1p4096_fwd[p];
        w3p4096_fwd[p] = w1p4096_fwd[p] * w2p4096_fwd[p];
    }
    for (int p = 0; p < 1024; p+=2) {
        w1p4096_fwd_avx[p/2] = _mm_loadu_ps(&(w1p4096_fwd[p].Re));
        w2p4096_fwd_avx[p/2] = _mm_loadu_ps(&(w2p4096_fwd[p].Re));
        w3p4096_fwd_avx[p/2] = _mm_loadu_ps(&(w3p4096_fwd[p].Re));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_8points(int N, complex_t<T> *x)
{
    const __m128 zm = {0.0, -0.0, 0.0, -0.0};
    const __m128 zm1 = {0.0, 0.0, 0.0, -0.0};

    __m128 x01 = _mm_loadu_ps(&(x[0].Re));
    __m128 x23 = _mm_loadu_ps(&(x[2].Re));
    __m128 x45 = _mm_loadu_ps(&(x[4].Re));
    __m128 x67 = _mm_loadu_ps(&(x[6].Re));
    __m128 a1 = _mm_add_ps(x01, x45);
    __m128 a2 = _mm_add_ps(x23, x67);
    __m128 a3 = _mm_sub_ps(x01, x45);
    __m128 xmy = _mm_xor_ps(zm, _mm_sub_ps(x23, x67));
    __m128 a4 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 0, 1));

    __m128 pm_a1 = _mm_add_ps(a1, a2);
    __m128 pm_a2 = v8xpz_f(_mm_add_ps(a3, a4));
    xmy = _mm_xor_ps(zm1, _mm_sub_ps(a1, a2));
    __m128 pm_a3 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 1, 0));
    __m128 pm_a4 = w8xpz_f(_mm_sub_ps(a3, a4));

    __m128 res1 = _mm_shuffle_ps(pm_a1, pm_a4, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 res2 = _mm_shuffle_ps(pm_a1, pm_a4, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 res3 = _mm_shuffle_ps(pm_a3, pm_a2, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 res4 = _mm_shuffle_ps(pm_a3, pm_a2, _MM_SHUFFLE(3, 2, 3, 2));

    _mm_storeu_ps(&(x[0].Re), _mm_add_ps(res1, res2));
    _mm_storeu_ps(&(x[2].Re), _mm_sub_ps(res3, res4));
    _mm_storeu_ps(&(x[4].Re), _mm_sub_ps(res1, res2));
    _mm_storeu_ps(&(x[6].Re), _mm_add_ps(res3, res4));
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_16points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[16];

    for (int p = 0; p < 4; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[4].Re));
        __m128 c = _mm_loadu_ps(&(x_p[8].Re));
        __m128 d = _mm_loadu_ps(&(x_p[12].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p16_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p16_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p16_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    __m256 a = _mm256_loadu_ps(&(y[0].Re));
    __m256 b = _mm256_loadu_ps(&(y[4].Re));
    __m256 c = _mm256_loadu_ps(&(y[8].Re));
    __m256 d = _mm256_loadu_ps(&(y[12].Re));
    __m256 apc = _mm256_add_ps(a, c);
    __m256 amc = _mm256_sub_ps(a, c);
    __m256 bpd = _mm256_add_ps(b, d);
    __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

    _mm256_storeu_ps(&(x[0].Re), _mm256_add_ps(apc, bpd));
    _mm256_storeu_ps(&(x[4].Re),  _mm256_sub_ps(amc, jbmd));
    _mm256_storeu_ps(&(x[8].Re), _mm256_sub_ps(apc, bpd));
    _mm256_storeu_ps(&(x[12].Re), _mm256_add_ps(amc, jbmd));
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_32points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[32];

    for (int p = 0; p < 8; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[8].Re));
        __m128 c = _mm_loadu_ps(&(x_p[16].Re));
        __m128 d = _mm_loadu_ps(&(x_p[24].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p32_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p32_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p32_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 8, s = 4
    for (int p = 0; p < 2; p++) {
        uint8_t sp = 4 * p;
        uint8_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p8_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p8_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p8_fwd_nse[p]);
        // The 32 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[8].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[16].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[24].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    for (int q = 0; q < 16; q+=4) {
        complex_t<T> *xq = x + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[16].Re));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(xq[16].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_64points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[64];

    for (int p = 0; p < 16; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[16].Re));
        __m128 c = _mm_loadu_ps(&(x_p[32].Re));
        __m128 d = _mm_loadu_ps(&(x_p[48].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p64_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p64_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p64_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 16, s = 4
    for (int p = 0; p < 4; p++) {
        uint8_t sp = 4 * p;
        uint8_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p16_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p16_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p16_fwd_nse[p]);
        // The 64 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[16].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[32].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[48].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    for (int q = 0; q < 16; q+=4) {
        complex_t<T> *xq = x + q;
        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[16].Re));
        __m256 c = _mm256_loadu_ps(&(xq[32].Re));
        __m256 d = _mm256_loadu_ps(&(xq[48].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[16].Re), _mm256_sub_ps(amc, jbmd));
        _mm256_storeu_ps(&(xq[32].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[48].Re), _mm256_add_ps(amc, jbmd));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_128points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[128];

    for (int p = 0; p < 32; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[32].Re));
        __m128 c = _mm_loadu_ps(&(x_p[64].Re));
        __m128 d = _mm_loadu_ps(&(x_p[96].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p128_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p128_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p128_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 32, s = 4
    for (int p = 0; p < 8; p++) {
        uint8_t sp = 4 * p;
        uint8_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p32_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p32_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p32_fwd_nse[p]);
        // The 128 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[32].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[64].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[96].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    // n = 8, s = 16
    for (int p = 0; p < 2; p++) {
        uint8_t sp = 16 * p;
        uint8_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p8_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p8_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p8_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[32].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[64].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[96].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 64; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(yq[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq[64].Re));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(xq[64].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_256points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[256];

    for (int p = 0; p < 64; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[64].Re));
        __m128 c = _mm_loadu_ps(&(x_p[128].Re));
        __m128 d = _mm_loadu_ps(&(x_p[192].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p256_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p256_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p256_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 64, s = 4
    for (int p = 0; p < 16; p++) {
        uint8_t sp = 4 * p;
        uint8_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p64_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p64_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p64_fwd_nse[p]);
        // The 256 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[64].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[128].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[192].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    // n = 16, s = 16
    for (int p = 0; p < 4; p++) {
        uint8_t sp = 16 * p;
        uint8_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p16_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p16_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p16_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[64].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[128].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[192].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 64; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(yq[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq[64].Re));
        __m256 c = _mm256_loadu_ps(&(yq[128].Re));
        __m256 d = _mm256_loadu_ps(&(yq[192].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[64].Re), _mm256_sub_ps(amc, jbmd));
        _mm256_storeu_ps(&(xq[128].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[192].Re), _mm256_add_ps(amc, jbmd));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_512points(int N, complex_t<T> *x)
{
    uint8_t wt_index = 0;
    complex_t<T> y[512];

    for (int p = 0; p < 128; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[128].Re));
        __m128 c = _mm_loadu_ps(&(x_p[256].Re));
        __m128 d = _mm_loadu_ps(&(x_p[384].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p512_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p512_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p512_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 128, s = 4
    for (int p = 0; p < 32; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p128_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p128_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p128_fwd_nse[p]);
        // The 512 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[128].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[256].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[384].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    // n = 32, s = 16
    for (int p = 0; p < 8; p++) {
        uint16_t sp = 16 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p32_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p32_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p32_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[128].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[256].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[384].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    // n = 8, s = 64
    for (int p = 0; p < 2; p++) {
        uint16_t sp = 64 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p8_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p8_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p8_fwd_nse[p]);
        for (int q = 0; q < 64; q+=4) {
            complex_t<T> *yq_sp = y + q + sp;
            complex_t<T> *xq_s4p = x + q + s4p;
            __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(yq_sp[128].Re));
            __m256 c = _mm256_loadu_ps(&(yq_sp[256].Re));
            __m256 d = _mm256_loadu_ps(&(yq_sp[384].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(xq_s4p[64].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(xq_s4p[128].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(xq_s4p[192].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 256; q+=4) {
        complex_t<T> *xq = x + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[256].Re));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(xq[256].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_1024points(int N, complex_t<T> *x)
{
    uint16_t wt_index = 0;
    complex_t<T> y[1024];

    for (int p = 0; p < 256; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[256].Re));
        __m128 c = _mm_loadu_ps(&(x_p[512].Re));
        __m128 d = _mm_loadu_ps(&(x_p[768].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p1024_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p1024_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p1024_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 256, s = 4
    for (int p = 0; p < 64; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p256_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p256_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p256_fwd_nse[p]);
        // The 1024 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[256].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[512].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[768].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    // n = 64, s = 16
    for (int p = 0; p < 16; p++) {
        uint16_t sp = 16 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p64_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p64_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p64_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[256].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[512].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[768].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    // n = 16, s = 64
    for (int p = 0; p < 4; p++) {
        uint16_t sp = 64 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p16_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p16_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p16_fwd_nse[p]);
        for (int q = 0; q < 64; q+=4) {
            complex_t<T> *yq_sp = y + q + sp;
            complex_t<T> *xq_s4p = x + q + s4p;
            __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(yq_sp[256].Re));
            __m256 c = _mm256_loadu_ps(&(yq_sp[512].Re));
            __m256 d = _mm256_loadu_ps(&(yq_sp[768].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(xq_s4p[64].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(xq_s4p[128].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(xq_s4p[192].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 256; q+=4) {
        complex_t<T> *xq = x + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[256].Re));
        __m256 c = _mm256_loadu_ps(&(xq[512].Re));
        __m256 d = _mm256_loadu_ps(&(xq[768].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[256].Re), _mm256_sub_ps(amc, jbmd));
        _mm256_storeu_ps(&(xq[512].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[768].Re), _mm256_add_ps(amc, jbmd));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_2048points(int N, complex_t<T> *x)
{
    uint16_t wt_index = 0;
    complex_t<T> y[2048];

    for (int p = 0; p < 512; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[512].Re));
        __m128 c = _mm_loadu_ps(&(x_p[1024].Re));
        __m128 d = _mm_loadu_ps(&(x_p[1536].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p2048_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p2048_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p2048_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }

    // n = 512, s = 4
    for (int p = 0; p < 128; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p512_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p512_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p512_fwd_nse[p]);
        // The 2048 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[512].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[1024].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[1536].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }

    // n = 128, s = 16
    for (int p = 0; p < 32; p++) {
        uint16_t sp = 16 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p128_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p128_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p128_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[512].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[1024].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[1536].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    // n = 32, s = 64
    for (int p = 0; p < 8; p++) {
        uint16_t sp = 64 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p32_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p32_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p32_fwd_nse[p]);
        for (int q = 0; q < 64; q+=4) {
            complex_t<T> *yq_sp = y + q + sp;
            complex_t<T> *xq_s4p = x + q + s4p;
            __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(yq_sp[512].Re));
            __m256 c = _mm256_loadu_ps(&(yq_sp[1024].Re));
            __m256 d = _mm256_loadu_ps(&(yq_sp[1536].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(xq_s4p[64].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(xq_s4p[128].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(xq_s4p[192].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }
    // n = 8, s = 256
    for (int p = 0; p < 2; p++) {
        uint16_t sp = 256 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p8_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p8_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p8_fwd_nse[p]);
        for (int q = 0; q < 256; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[512].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[1024].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[1536].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[256].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[512].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[768].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 1024; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(yq[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq[1024].Re));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(xq[1024].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_4096points(int N, complex_t<T> *x)
{
    uint16_t wt_index = 0;
    complex_t<T> y[4096];

    for (int p = 0; p < 1024; p+=2, wt_index+=1) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[1024].Re));
        __m128 c = _mm_loadu_ps(&(x_p[2048].Re));
        __m128 d = _mm_loadu_ps(&(x_p[3072].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p4096_fwd_avx[wt_index], _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p4096_fwd_avx[wt_index], _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p4096_fwd_avx[wt_index], _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }
    // n = 1024, s = 4
    for (int p = 0; p < 256; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *yq_sp = y + sp;
        complex_t<T> *xq_s4p = x + s4p;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p1024_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p1024_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p1024_fwd_nse[p]);
        // The 4096 points case : q = 1
        __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq_sp[1024].Re));
        __m256 c = _mm256_loadu_ps(&(yq_sp[2048].Re));
        __m256 d = _mm256_loadu_ps(&(yq_sp[3072].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(xq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(xq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }
    // n = 256, s = 16
    for (int p = 0; p < 64; p++) {
        uint16_t sp = 16 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p256_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p256_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p256_fwd_nse[p]);
        for (int q = 0; q < 16; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[1024].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[2048].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[3072].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[16].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[32].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[48].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }
    // n = 64, s = 64
    for (int p = 0; p < 16; p++) {
        uint16_t sp = 64 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p64_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p64_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p64_fwd_nse[p]);
        for (int q = 0; q < 64; q+=4) {
            complex_t<T> *yq_sp = y + q + sp;
            complex_t<T> *xq_s4p = x + q + s4p;
            __m256 a = _mm256_loadu_ps(&(yq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(yq_sp[1024].Re));
            __m256 c = _mm256_loadu_ps(&(yq_sp[2048].Re));
            __m256 d = _mm256_loadu_ps(&(yq_sp[3072].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(xq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(xq_s4p[64].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(xq_s4p[128].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(xq_s4p[192].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }
    // n = 16, s = 256
    for (int p = 0; p < 4; p++) {
        uint16_t sp = 256 * p;
        uint16_t s4p = 4 * sp;
        // Set the weight
        __m256 w1p = duppz2_lo(w1p16_fwd_nse[p]);
        __m256 w2p = duppz2_lo(w2p16_fwd_nse[p]);
        __m256 w3p = duppz2_lo(w3p16_fwd_nse[p]);
        for (int q = 0; q < 256; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[1024].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[2048].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[3072].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[256].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[512].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[768].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }

    for (int q = 0; q < 1024; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(yq[0].Re));
        __m256 b = _mm256_loadu_ps(&(yq[1024].Re));
        __m256 c = _mm256_loadu_ps(&(yq[2048].Re));
        __m256 d = _mm256_loadu_ps(&(yq[3072].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(xq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[1024].Re), _mm256_sub_ps(amc, jbmd));
        _mm256_storeu_ps(&(xq[2048].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(xq[3072].Re), _mm256_add_ps(amc, jbmd));
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_fft_avx_whole<T>::my_fft(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    if (sizeof(T) != 4) {
        fprintf(stderr, "ONLY SUPPORT FLOAT SIMD FFT NOW !\n");
        exit(-1);
    }

    switch (N)
    {
    case 1 : {
        } return;
    case 2 : {
            // N = 2 is treated as the special situation, the type of float32_t data.
            __m128 a = _mm_loadu_ps(&x[0].Re);
            __m128 r = _mm_addsub_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 3, 1)),
                                     _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 2, 1, 3)));
            _mm_storeu_ps(&x[0].Re, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 2, 1, 3)));
        } break;
    case 4 : {
            const __m128 zm = {0.0, 0.0, 0.0, -0.0};
            __m128 ab = _mm_loadu_ps(&x[0].Re);
            __m128 cd = _mm_loadu_ps(&x[2].Re);
            __m128 r1 = _mm_add_ps(ab, cd);
            __m128 xmy = _mm_xor_ps(zm, _mm_sub_ps(ab, cd));
            __m128 r2 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 1, 0));

            __m128 res1 = _mm_addsub_ps(_mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 0, 3, 1)),
                                        _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 2, 1, 3)));
            __m128 res2 = _mm_addsub_ps(_mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 0, 3, 1)),
                                        _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(0, 2, 1, 3)));
            _mm_storeu_ps(&x[0].Re, _mm_shuffle_ps(res1, res2, _MM_SHUFFLE(0, 2, 1, 3)));
            _mm_storeu_ps(&x[2].Re, _mm_shuffle_ps(res1, res2, _MM_SHUFFLE(1, 3, 0, 2)));
        } break;
    case 8 : {
            my_fft_8points(N, x);
        } break;
    case 16: {
            // The member function
            my_fft_16points(N, x);
        } break;
    case 32 : {
            // The member function
            my_fft_32points(N, x);
        } break;
    case 64 : {
            // The member function
            my_fft_64points(N, x);
        } break;
    case 128 : {
            // The member function
            my_fft_128points(N, x);
        } break;
    case 256 : {
            // The member function
            my_fft_256points(N, x);
        } break;
    case 512 : {
            // The member function
            my_fft_512points(N, x);
        } break;
    case 1024 : {
            // The member function
            my_fft_1024points(N, x);
        } break;
    case 2048 : {
            // The member function
            my_fft_2048points(N, x);
        } break;
    case 4096 : {
            // The member function
            my_fft_4096points(N, x);
        } break;
    default : {
            // The other situation
            fprintf(stderr, "NOT SUPPORTED POINTS !\n");
        } break;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_8points(int N, complex_t<T> *x)
{
    const __m128 zm = {0.0, -0.0, 0.0, -0.0};
    const __m128 zm1 = {0.0, 0.0, 0.0, -0.0};
    const __m128 eight = {8.0, 8.0, 8.0, 8.0};
    __m128 x01 = _mm_loadu_ps(&(x[0].Re));
    __m128 x23 = _mm_loadu_ps(&(x[2].Re));
    __m128 x45 = _mm_loadu_ps(&(x[4].Re));
    __m128 x67 = _mm_loadu_ps(&(x[6].Re));
    __m128 a1 = _mm_add_ps(x01, x45);
    __m128 a2 = _mm_add_ps(x23, x67);
    __m128 a3 = _mm_sub_ps(x01, x45);
    __m128 xmy = _mm_xor_ps(zm, _mm_sub_ps(x23, x67));
    __m128 a4 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 0, 1));

    __m128 pm_a1 = _mm_add_ps(a1, a2);
    __m128 pm_a2 = v8xpz_f(_mm_add_ps(a3, a4));
    xmy = _mm_xor_ps(zm1, _mm_sub_ps(a1, a2));
    __m128 pm_a3 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 1, 0));
    __m128 pm_a4 = w8xpz_f(_mm_sub_ps(a3, a4));

    __m128 res1 = _mm_shuffle_ps(pm_a1, pm_a2, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 res2 = _mm_shuffle_ps(pm_a1, pm_a2, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 res3 = _mm_shuffle_ps(pm_a3, pm_a4, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 res4 = _mm_shuffle_ps(pm_a3, pm_a4, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 res5 = _mm_add_ps(res3, res4);
    __m128 res6 = _mm_sub_ps(res3, res4);

    _mm_storeu_ps(&(x[0].Re), _mm_add_ps(res1, res2));
    _mm_storeu_ps(&(x[2].Re), _mm_shuffle_ps(res5, res6, _MM_SHUFFLE(3, 2, 1, 0)));
    _mm_storeu_ps(&(x[4].Re), _mm_sub_ps(res1, res2));
    _mm_storeu_ps(&(x[6].Re), _mm_shuffle_ps(res6, res5, _MM_SHUFFLE(3, 2, 1, 0)));
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_16points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[16];
    for (int p = 0; p < 4; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 4];
        const complex_t<T> c = x[p + 8];
        const complex_t<T> d = x[p + 12];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p] = apc + bpd;
        y[4*p + 1] = conj(w1p16_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p16_fwd[p])*(apc - bpd);
        y[4*p + 3] = conj(w3p16_fwd[p])*(amc - jbmd);
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
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_32points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[32];
    for (int p = 0; p < 8; p++) {
        // Pay attention, sin(theta0
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 8];
        const complex_t<T> c = x[p + 16];
        const complex_t<T> d = x[p + 24];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p + 0] =      apc +  bpd;
        y[4*p + 1] = conj(w1p32_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p32_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p32_fwd[p])*(amc - jbmd);
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
        x[q + 20] = conj(w1p8_fwd)*(amc + jbmd);
        x[q + 24] = conj(w2p8_fwd)*(apc -  bpd);
        x[q + 28] = conj(w3p8_fwd)*(amc - jbmd);
    }

    for (int q = 0; q < 16; q++) {
        const complex_t<T> a = x[q];
        const complex_t<T> b = x[q + 16];
        x[q + 0]  = a + b;
        x[q + 16] = a - b;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_64points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[64];
    for (int p = 0; p < 16; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 16];
        const complex_t<T> c = x[p + 32];
        const complex_t<T> d = x[p + 48];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p + 0] =      apc +  bpd;
        y[4*p + 1] = conj(w1p64_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p64_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p64_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*p + 16];
            const complex_t<T> c = y[q + 4*p + 32];
            const complex_t<T> d = y[q + 4*p + 48];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 16*p] =      apc +  bpd;
            x[q + 16*p + 4] = conj(w1p16_fwd[p])*(amc + jbmd);
            x[q + 16*p + 8] = conj(w2p16_fwd[p])*(apc -  bpd);
            x[q + 16*p + 12] = conj(w3p16_fwd[p])*(amc - jbmd);
        }
    }

    for (int q = 0; q < 16; q++) {
        const complex_t<T> a = x[q];
        const complex_t<T> b = x[q + 16];
        const complex_t<T> c = x[q + 32];
        const complex_t<T> d = x[q + 48];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q + 0]  = apc +  bpd;
        x[q + 16] = amc + jbmd;
        x[q + 32] = apc - bpd;
        x[q + 48] = amc - jbmd;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_128points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);
    const T theta1 = 2*M_PI/32;
    const T theta2 = 2*M_PI/8;

    complex_t<T> y[128];
    for (int p = 0; p < 32; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 32];
        const complex_t<T> c = x[p + 64];
        const complex_t<T> d = x[p + 96];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p] =      apc +  bpd;
        y[4*p + 1] = conj(w1p128_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p128_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p128_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 8; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*p + 32];
            const complex_t<T> c = y[q + 4*p + 64];
            const complex_t<T> d = y[q + 4*p + 96];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 16*p] =      apc +  bpd;
            x[q + 16*p + 4] = conj(w1p32_fwd[p])*(amc + jbmd);
            x[q + 16*p + 8] = conj(w2p32_fwd[p])*(apc - bpd);
            x[q + 16*p + 12] = conj(w3p32_fwd[p])*(amc - jbmd);
        }
    }

    // p = 0
    for (int q = 0; q < 16; q++) {
        const complex_t<T> a = x[q];
        const complex_t<T> b = x[q + 32];
        const complex_t<T> c = x[q + 64];
        const complex_t<T> d = x[q + 96];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[q] =      apc +  bpd;
        y[q + 16] = amc + jbmd;
        y[q + 32] = apc -  bpd;
        y[q + 48] = amc - jbmd;
    }
    // p = 1
    for (int q = 0; q < 16; q++) {
        const complex_t<T> a = x[q + 16];
        const complex_t<T> b = x[q + 48];
        const complex_t<T> c = x[q + 80];
        const complex_t<T> d = x[q + 112];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[q + 64] =      apc +  bpd;
        y[q + 80] = conj(w1p8_fwd)*(amc + jbmd);
        y[q + 96] = conj(w2p8_fwd)*(apc -  bpd);
        y[q + 112] = conj(w3p8_fwd)*(amc - jbmd);
    }

    for (int q = 0; q < 64; q++) {
        const complex_t<T> a = y[q + 0];
        const complex_t<T> b = y[q + 64];
        x[q] = a + b;
        x[q + 64] = a - b;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_256points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[256];
    for (int p = 0; p < 64; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 64];
        const complex_t<T> c = x[p + 128];
        const complex_t<T> d = x[p + 192];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p] = apc +  bpd;
        y[4*p + 1] = conj(w1p256_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p256_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p256_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 16; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*(p + 16)];
            const complex_t<T> c = y[q + 4*(p + 32)];
            const complex_t<T> d = y[q + 4*(p + 48)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 16*p] =      apc +  bpd;
            x[q + 16*p + 4] = conj(w1p64_fwd[p])*(amc + jbmd);
            x[q + 16*p + 8] = conj(w2p64_fwd[p])*(apc -  bpd);
            x[q + 16*p + 12] = conj(w3p64_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 16; q++) {
            const complex_t<T> a = x[q + 16*p];
            const complex_t<T> b = x[q + 16*(p + 4)];
            const complex_t<T> c = x[q + 16*(p + 8)];
            const complex_t<T> d = x[q + 16*(p + 12)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 64*p] =      apc + bpd;
            y[q + 64*p + 16] = conj(w1p16_fwd[p])*(amc + jbmd);
            y[q + 64*p + 32] = conj(w2p16_fwd[p])*(apc - bpd);
            y[q + 64*p + 48] = conj(w3p16_fwd[p])*(amc - jbmd);
        }
    }

    for (int q = 0; q < 64; q++) {
        const complex_t<T> a = y[q];
        const complex_t<T> b = y[q + 64];
        const complex_t<T> c = y[q + 128];
        const complex_t<T> d = y[q + 192];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q]   = apc + bpd;
        x[q + 64]   = amc + jbmd;
        x[q + 128] = apc - bpd;
        x[q + 192] = amc - jbmd;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_512points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[512];
    for (int p = 0; p < 128; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 128];
        const complex_t<T> c = x[p + 256];
        const complex_t<T> d = x[p + 384];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p] = apc +  bpd;
        y[4*p + 1] = conj(w1p512_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p512_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p512_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 32; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*(p + 32)];
            const complex_t<T> c = y[q + 4*(p + 64)];
            const complex_t<T> d = y[q + 4*(p + 96)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 4*(4*p + 0)] =      apc +  bpd;
            x[q + 4*(4*p + 1)] = conj(w1p128_fwd[p])*(amc + jbmd);
            x[q + 4*(4*p + 2)] = conj(w2p128_fwd[p])*(apc -  bpd);
            x[q + 4*(4*p + 3)] = conj(w3p128_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 8; p++) {
        for (int q = 0; q < 16; q++) {
            const complex_t<T> a = x[q + 16*p];
            const complex_t<T> b = x[q + 16*(p + 8)];
            const complex_t<T> c = x[q + 16*(p + 16)];
            const complex_t<T> d = x[q + 16*(p + 24)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 16*(4*p + 0)] =      apc +  bpd;
            y[q + 16*(4*p + 1)] = conj(w1p32_fwd[p])*(amc + jbmd);
            y[q + 16*(4*p + 2)] = conj(w2p32_fwd[p])*(apc -  bpd);
            y[q + 16*(4*p + 3)] = conj(w3p32_fwd[p])*(amc - jbmd);
        }
    }

    // p = 0
    for (int q = 0; q < 64; q++) {
        const complex_t<T> a = y[q];
        const complex_t<T> b = y[q + 128];
        const complex_t<T> c = y[q + 256];
        const complex_t<T> d = y[q + 384];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q] =      apc +  bpd;
        x[q + 64] = amc + jbmd;
        x[q + 128] = apc -  bpd;
        x[q + 192] = amc - jbmd;
    }
    // p = 1
    for (int q = 0; q < 64; q++) {
        const complex_t<T> a = y[q + 64];
        const complex_t<T> b = y[q + 192];
        const complex_t<T> c = y[q + 320];
        const complex_t<T> d = y[q + 448];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q + 256] =      apc +  bpd;
        x[q + 320] = conj(w1p8_fwd)*(amc + jbmd);
        x[q + 384] = conj(w2p8_fwd)*(apc -  bpd);
        x[q + 448] = conj(w3p8_fwd)*(amc - jbmd);
    }

    for (int q = 0; q < 256; q++) {
        const complex_t<T> a = x[q + 0];
        const complex_t<T> b = x[q + 256];
        x[q + 0] = a + b;
        x[q + 256] = a - b;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_1024points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[1024];
    for (int p = 0; p < 256; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 256];
        const complex_t<T> c = x[p + 512];
        const complex_t<T> d = x[p + 768];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p + 0] =      apc +  bpd;
        y[4*p + 1] = conj(w1p1024_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p1024_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p1024_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 64; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*(p + 64)];
            const complex_t<T> c = y[q + 4*(p + 128)];
            const complex_t<T> d = y[q + 4*(p + 192)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 16*p] =      apc +  bpd;
            x[q + 16*p + 4] = conj(w1p256_fwd[p])*(amc + jbmd);
            x[q + 16*p + 8] = conj(w2p256_fwd[p])*(apc -  bpd);
            x[q + 16*p + 12] = conj(w3p256_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 16; p++) {
        for (int q = 0; q < 16; q++) {
            const complex_t<T> a = x[q + 16*p];
            const complex_t<T> b = x[q + 16*(p + 16)];
            const complex_t<T> c = x[q + 16*(p + 32)];
            const complex_t<T> d = x[q + 16*(p + 48)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 64*p] =      apc +  bpd;
            y[q + 64*p + 16] = conj(w1p64_fwd[p])*(amc + jbmd);
            y[q + 64*p + 32] = conj(w2p64_fwd[p])*(apc -  bpd);
            y[q + 64*p + 48] = conj(w3p64_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 64; q++) {
            const complex_t<T> a = y[q + 64*p];
            const complex_t<T> b = y[q + 64*(p + 4)];
            const complex_t<T> c = y[q + 64*(p + 8)];
            const complex_t<T> d = y[q + 64*(p + 12)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 256*p] =      apc +  bpd;
            x[q + 256*p + 64] = conj(w1p16_fwd[p])*(amc + jbmd);
            x[q + 256*p + 128] = conj(w2p16_fwd[p])*(apc -  bpd);
            x[q + 256*p + 192] = conj(w3p16_fwd[p])*(amc - jbmd);
        }
    }

    for (int q = 0; q < 256; q++) {
        const complex_t<T> a = x[q];
        const complex_t<T> b = x[q + 256];
        const complex_t<T> c = x[q + 512];
        const complex_t<T> d = x[q + 768];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q]   = apc + bpd;
        x[q + 256]   = amc + jbmd;
        x[q + 512] = apc - bpd;
        x[q + 768] = amc - jbmd;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_2048points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[2048];

    for (int p = 0; p < 512; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 512];
        const complex_t<T> c = x[p + 1024];
        const complex_t<T> d = x[p + 1536];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p + 0] =      apc +  bpd;
        y[4*p + 1] = conj(w1p2048_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p2048_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p2048_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 128; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*(p + 128)];
            const complex_t<T> c = y[q + 4*(p + 256)];
            const complex_t<T> d = y[q + 4*(p + 384)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 16*p] =      apc +  bpd;
            x[q + 16*p + 4] = conj(w1p512_fwd[p])*(amc + jbmd);
            x[q + 16*p + 8] = conj(w2p512_fwd[p])*(apc -  bpd);
            x[q + 16*p + 12] = conj(w3p512_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 32; p++) {
        for (int q = 0; q < 16; q++) {
            const complex_t<T> a = x[q + 16*p];
            const complex_t<T> b = x[q + 16*(p + 32)];
            const complex_t<T> c = x[q + 16*(p + 64)];
            const complex_t<T> d = x[q + 16*(p + 96)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 64*p] =      apc +  bpd;
            y[q + 64*p + 16] = conj(w1p128_fwd[p])*(amc + jbmd);
            y[q + 64*p + 32] = conj(w2p128_fwd[p])*(apc -  bpd);
            y[q + 64*p + 48] = conj(w3p128_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 8; p++) {
        for (int q = 0; q < 64; q++) {
            const complex_t<T> a = y[q + 64*p];
            const complex_t<T> b = y[q + 64*(p + 8)];
            const complex_t<T> c = y[q + 64*(p + 16)];
            const complex_t<T> d = y[q + 64*(p + 24)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 64*(4*p + 0)] =      apc +  bpd;
            x[q + 64*(4*p + 1)] = conj(w1p32_fwd[p])*(amc + jbmd);
            x[q + 64*(4*p + 2)] = conj(w2p32_fwd[p])*(apc -  bpd);
            x[q + 64*(4*p + 3)] = conj(w3p32_fwd[p])*(amc - jbmd);
        }
    }

    // p = 0
    for (int q = 0; q < 256; q++) {
        const complex_t<T> a = x[q];
        const complex_t<T> b = x[q + 512];
        const complex_t<T> c = x[q + 1024];
        const complex_t<T> d = x[q + 1536];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[q] =      apc +  bpd;
        y[q + 256] = amc + jbmd;
        y[q + 512] = apc -  bpd;
        y[q + 768] = amc - jbmd;
    }
    // p = 1
    for (int q = 0; q < 256; q++) {
        const complex_t<T> a = x[q + 256];
        const complex_t<T> b = x[q + 768];
        const complex_t<T> c = x[q + 1280];
        const complex_t<T> d = x[q + 1792];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[q + 1024] =      apc +  bpd;
        y[q + 1280] = conj(w1p8_fwd)*(amc + jbmd);
        y[q + 1536] = conj(w2p8_fwd)*(apc -  bpd);
        y[q + 1792] = conj(w3p8_fwd)*(amc - jbmd);
    }

    for (int q = 0; q < 1024; q++) {
        const complex_t<T> a = y[q];
        const complex_t<T> b = y[q + 1024];
        x[q] = a + b;
        x[q + 1024] = a - b;
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_4096points(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    complex_t<T> y[4096];
    for (int p = 0; p < 1024; p++) {
        const complex_t<T> a = x[p];
        const complex_t<T> b = x[p + 1024];
        const complex_t<T> c = x[p + 2048];
        const complex_t<T> d = x[p + 3072];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        y[4*p] =      apc +  bpd;
        y[4*p + 1] = conj(w1p4096_fwd[p])*(amc + jbmd);
        y[4*p + 2] = conj(w2p4096_fwd[p])*(apc -  bpd);
        y[4*p + 3] = conj(w3p4096_fwd[p])*(amc - jbmd);
    }

    for (int p = 0; p < 256; p++) {
        for (int q = 0; q < 4; q++) {
            const complex_t<T> a = y[q + 4*p];
            const complex_t<T> b = y[q + 4*(p + 256)];
            const complex_t<T> c = y[q + 4*(p + 512)];
            const complex_t<T> d = y[q + 4*(p + 768)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 4*(4*p + 0)] =      apc +  bpd;
            x[q + 4*(4*p + 1)] = conj(w1p1024_fwd[p])*(amc + jbmd);
            x[q + 4*(4*p + 2)] = conj(w2p1024_fwd[p])*(apc -  bpd);
            x[q + 4*(4*p + 3)] = conj(w3p1024_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 64; p++) {
        for (int q = 0; q < 16; q++) {
            const complex_t<T> a = x[q + 16*p];
            const complex_t<T> b = x[q + 16*(p + 64)];
            const complex_t<T> c = x[q + 16*(p + 128)];
            const complex_t<T> d = x[q + 16*(p + 192)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 16*(4*p + 0)] =      apc +  bpd;
            y[q + 16*(4*p + 1)] = conj(w1p256_fwd[p])*(amc + jbmd);
            y[q + 16*(4*p + 2)] = conj(w2p256_fwd[p])*(apc -  bpd);
            y[q + 16*(4*p + 3)] = conj(w3p256_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 16; p++) {
        for (int q = 0; q < 64; q++) {
            const complex_t<T> a = y[q + 64*p];
            const complex_t<T> b = y[q + 64*(p + 16)];
            const complex_t<T> c = y[q + 64*(p + 32)];
            const complex_t<T> d = y[q + 64*(p + 48)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            x[q + 64*(4*p + 0)] =      apc +  bpd;
            x[q + 64*(4*p + 1)] = conj(w1p64_fwd[p])*(amc + jbmd);
            x[q + 64*(4*p + 2)] = conj(w2p64_fwd[p])*(apc -  bpd);
            x[q + 64*(4*p + 3)] = conj(w3p64_fwd[p])*(amc - jbmd);
        }
    }

    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 256; q++) {
            const complex_t<T> a = x[q + 256*p];
            const complex_t<T> b = x[q + 256*(p + 4)];
            const complex_t<T> c = x[q + 256*(p + 8)];
            const complex_t<T> d = x[q + 256*(p + 12)];
            const complex_t<T>  apc =    a + c;
            const complex_t<T>  amc =    a - c;
            const complex_t<T>  bpd =    b + d;
            const complex_t<T> jbmd = j*(b - d);
            y[q + 256*(4*p + 0)] = apc + bpd;
            y[q + 256*(4*p + 1)] = conj(w1p16_fwd[p])*(amc + jbmd);
            y[q + 256*(4*p + 2)] = conj(w2p16_fwd[p])*(apc -  bpd);
            y[q + 256*(4*p + 3)] = conj(w3p16_fwd[p])*(amc - jbmd);
        }
    }

    for (int q = 0; q < 1024; q++) {
        const complex_t<T> a = y[q];
        const complex_t<T> b = y[q + 1024];
        const complex_t<T> c = y[q + 2048];
        const complex_t<T> d = y[q + 3072];
        const complex_t<T>  apc =    a + c;
        const complex_t<T>  amc =    a - c;
        const complex_t<T>  bpd =    b + d;
        const complex_t<T> jbmd = j*(b - d);
        x[q] = apc + bpd;
        x[q + 1024] = amc + jbmd;
        x[q + 2048] = apc - bpd;
        x[q + 3072] = amc - jbmd;
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
template<typename T>
void my_fft_avx_whole<T>::my_ifft(int N, complex_t<T> *x)
{
    static const complex_t<T> j = complex_t<T>(0, 1);

    if (sizeof(T) != 4) {
        fprintf(stderr, "ONLY SUPPORT FLOAT SIMD INVERSE-FFT NOW !\n");
        exit(-1);
    }

    switch (N)
    {
    case 1 : {
        } return;
    case 2 : {
            // N = 2 is treated as the special situation
            const __m128 two = {2.0, 2.0, 2.0, 2.0};
            // N = 2 is treated as the special situation, the type of float32_t data.
            __m128 a = _mm_loadu_ps(&x[0].Re);
            __m128 r = _mm_addsub_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 3, 1)),
                                     _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 2, 1, 3)));
            _mm_storeu_ps(&x[0].Re, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 2, 1, 3)));
        } break;
    case 4 : {
            const __m128 zm = {0.0, 0.0, 0.0, -0.0};
            const __m128 four = {4.0, 4.0, 4.0, 4.0};
            __m128 ab = _mm_loadu_ps(&x[0].Re);
            __m128 cd = _mm_loadu_ps(&x[2].Re);
            __m128 r1 = _mm_add_ps(ab, cd);
            __m128 xmy = _mm_xor_ps(zm, _mm_sub_ps(ab, cd));
            __m128 r2 = _mm_shuffle_ps(xmy, xmy, _MM_SHUFFLE(2, 3, 1, 0));

            __m128 res1 = _mm_addsub_ps(_mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 0, 3, 1)),
                                        _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 2, 1, 3)));
            __m128 res2 = _mm_addsub_ps(_mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 0, 3, 1)),
                                        _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(0, 2, 1, 3)));
            _mm_storeu_ps(&x[0].Re, _mm_shuffle_ps(res1, res2, _MM_SHUFFLE(1, 3, 1, 3)));
            _mm_storeu_ps(&x[2].Re, _mm_shuffle_ps(res1, res2, _MM_SHUFFLE(0, 2, 0, 2)));
        } break;
    case 8 : {
            my_ifft_8points(N, x);
        } break;
    case 16 : {
            // Member function
            my_ifft_16points(N, x);
        } break;
    case 32 : {
            // Member function
            my_ifft_32points(N, x);
        } break;
    case 64 : {
            // Member function
            my_ifft_64points(N, x);
        } break;
    case 128 : {
            // Member function
            my_ifft_128points(N, x);
        } break;
    case 256 : {
            // Member function
            my_ifft_256points(N, x);
        } break;
    case 512 : {
            // Member function
            my_ifft_512points(N, x);
        } break;
    case 1024 : {
            // Member function
            my_ifft_1024points(N, x);
        } break;
    case 2048 : {
            // Member function
            my_ifft_2048points(N, x);
        } break;
    case 4096 : {
            // Member function
            my_ifft_4096points(N, x);
        } break;
    default : {
            // The other situation
            fprintf(stderr, "NOT SUPPORTED POINTS !\n");
        } break;
    }
}

#endif
