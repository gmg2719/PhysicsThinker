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
//          Weight values are designed         //
/////////////////////////////////////////////////
inline uint8_t my_position_of_weight(int N)
{
    switch (N) {
    case 8:
        return 0;
    case 16:
        return 1;
    case 32:
        return 2;
    case 64:
        return 3;
    case 128:
        return 4;
    case 256:
        return 5;
    case 512:
        return 6;
    case 1024:
        return 7;
    case 2048:
        return 8;
    case 4096:
        return 9;
    default:
        return 0;
    }
}

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
    // Forward FFT usage
    complex_t<T> *w1p_fwd[10];
    complex_t<T> *w2p_fwd[10];
    complex_t<T> *w3p_fwd[10];

    // Backward FFT usage
    complex_t<T> *w1p_back[10];
    complex_t<T> *w2p_back[10];
    complex_t<T> *w3p_back[10];

    uint16_t wp_points[10] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069};

    // The construct interface
    my_fft_avx_whole(int N);
    // FFT forward operations
    inline void my_fft_first_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_fft_butterfly_s4(int N, complex_t<T> *x, complex_t<T> *y);
    inline void my_fft_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_fft_bottom2_butterfly(int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_fft_bottom4_butterfly(int s, complex_t<T> *x, complex_t<T> *y);
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
    inline void my_ifft_first_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_ifft_butterfly_s4(int N, complex_t<T> *x, complex_t<T> *y);
    inline void my_ifft_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_ifft_bottom2_butterfly(int s, complex_t<T> *x, complex_t<T> *y);
    inline void my_ifft_bottom4_butterfly(int s, complex_t<T> *x, complex_t<T> *y);
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

    // Provided wrapper interfaces
    void my_fft(int N, complex_t<T> *x);
    void my_ifft(int N, complex_t<T> *x);
};

template<typename T>
my_fft_avx_whole<T>::my_fft_avx_whole(int N)
{
    // Coefficents initialization
    for (uint8_t i = 0; i < 10; i++) {
        uint16_t points = wp_points[i];
        w1p_fwd[i] = new complex_t<T>[points/4];
        w2p_fwd[i] = new complex_t<T>[points/4];
        w3p_fwd[i] = new complex_t<T>[points/4];
        w1p_back[i] = new complex_t<T>[points/4];
        w2p_back[i] = new complex_t<T>[points/4];
        w3p_back[i] = new complex_t<T>[points/4];
        T theta = 2*M_PI/points;

        for (uint16_t p = 0; p < points/4; p++) {
            w1p_fwd[i][p] = complex_t<T>(cos(p*theta), -sin(p*theta));
            w2p_fwd[i][p] = w1p_fwd[i][p] * w1p_fwd[i][p];
            w3p_fwd[i][p] = w1p_fwd[i][p] * w2p_fwd[i][p];

            w1p_back[i][p] = complex_t<T>(cos(p*theta), sin(p*theta));
            w2p_back[i][p] = w1p_back[i][p] * w1p_back[i][p];
            w3p_back[i][p] = w1p_back[i][p] * w2p_back[i][p];
        }
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
inline void my_fft_avx_whole<T>::my_fft_first_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n1 = N/4;
    uint16_t n2 = N/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_fwd[index];
    complex_t<T> *w2 = w2p_fwd[index];
    complex_t<T> *w3 = w3p_fwd[index];

    for (uint16_t p = 0; p < n1; p+=2) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 w1p_avx = _mm_loadu_ps(&(w1[p].Re));
        __m128 w2p_avx = _mm_loadu_ps(&(w2[p].Re));
        __m128 w3p_avx = _mm_loadu_ps(&(w3[p].Re));

        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[n1].Re));
        __m128 c = _mm_loadu_ps(&(x_p[n2].Re));
        __m128 d = _mm_loadu_ps(&(x_p[n3].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p_avx, _mm_sub_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p_avx, _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p_avx, _mm_add_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
#if 0
        printf("p = %d\n", p);
        float ans[4];
        _mm_store_ps(ans, ab);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
        _mm_store_ps(ans, cd);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
        _mm_store_ps(ans, AB);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
#endif
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_butterfly_s4(int N, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n = N/4;
    uint16_t n1 = N;
    uint16_t n2 = 2*N;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_fwd[index];
    complex_t<T> *w2 = w2p_fwd[index];
    complex_t<T> *w3 = w3p_fwd[index];

    // s = 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *xq_sp = x + sp;
        complex_t<T> *yq_s4p = y + s4p;
        __m128 w1p_avx = _mm_set_ps(w1[p].Im, w1[p].Re, w1[p].Im, w1[p].Re);
        __m128 w2p_avx = _mm_set_ps(w2[p].Im, w2[p].Re, w2[p].Im, w2[p].Re);
        __m128 w3p_avx = _mm_set_ps(w3[p].Im, w3[p].Re, w3[p].Im, w3[p].Re);

        // Set the weight
        __m256 w1p = duppz2_lo(w1p_avx);
        __m256 w2p = duppz2_lo(w2p_avx);
        __m256 w3p = duppz2_lo(w3p_avx);

        // The points case : q = 1
        __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq_sp[n1].Re));
        __m256 c = _mm256_loadu_ps(&(xq_sp[n2].Re));
        __m256 d = _mm256_loadu_ps(&(xq_sp[n3].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

#if 0
        printf("q = %d %d %d %d\n", 0, n1, n2, n3);
        float ans[8];
        _mm256_store_ps(ans, mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                            ans[4], ans[5], ans[6], ans[7]);
        _mm256_store_ps(ans, mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                            ans[4], ans[5], ans[6], ans[7]);
        _mm256_store_ps(ans, mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                            ans[4], ans[5], ans[6], ans[7]);
#endif

        _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(yq_s4p[4].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
        _mm256_storeu_ps(&(yq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(yq_s4p[12].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n = N/4;
    uint16_t n1 = (N*s)/4;
    uint16_t n2 = (N*s)/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_fwd[index];
    complex_t<T> *w2 = w2p_fwd[index];
    complex_t<T> *w3 = w3p_fwd[index];

    // s != 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = s * p;
        uint16_t s4p = 4 * sp;
        __m128 w1p_avx = _mm_set_ps(w1[p].Im, w1[p].Re, w1[p].Im, w1[p].Re);
        __m128 w2p_avx = _mm_set_ps(w2[p].Im, w2[p].Re, w2[p].Im, w2[p].Re);
        __m128 w3p_avx = _mm_set_ps(w3[p].Im, w3[p].Re, w3[p].Im, w3[p].Re);
        // Set the weight
        __m256 w1p = duppz2_lo(w1p_avx);
        __m256 w2p = duppz2_lo(w2p_avx);
        __m256 w3p = duppz2_lo(w3p_avx);

        for (uint16_t q = 0; q < s; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[n1].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[n2].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[n3].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
#if 0
            printf("q = %d %d %d %d %d %d\n", q, n1, n2, n3, sp, s4p);
            float ans[8];
            _mm256_store_ps(ans, b);
            printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                                ans[4], ans[5], ans[6], ans[7]);
            _mm256_store_ps(ans, c);
            printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                                ans[4], ans[5], ans[6], ans[7]);
            _mm256_store_ps(ans, d);
            printf("%.8e %.8e %.8e %.8e \n%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3],
                                                                ans[4], ans[5], ans[6], ans[7]);
#endif

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[s].Re), mulpz2(w1p, _mm256_sub_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[s*2].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[s*3].Re), mulpz2(w3p, _mm256_add_ps(amc, jbmd)));
        }
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_bottom2_butterfly(int s, complex_t<T> *x, complex_t<T> *y)
{
    for (uint16_t q = 0; q < s; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_bottom4_butterfly(int s, complex_t<T> *x, complex_t<T> *y)
{
    for (int q = 0; q < s; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        __m256 c = _mm256_loadu_ps(&(xq[s*2].Re));
        __m256 d = _mm256_loadu_ps(&(xq[s*3].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_sub_ps(amc, jbmd));
        _mm256_storeu_ps(&(yq[s*2].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(yq[s*3].Re), _mm256_add_ps(amc, jbmd));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_16points(int N, complex_t<T> *x)
{
    complex_t<T> y[16];
    my_fft_first_butterfly(16, 1, x, y);
    my_fft_bottom4_butterfly(4, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_32points(int N, complex_t<T> *x)
{
    complex_t<T> y[32];
    my_fft_first_butterfly(32, 1, x, y);
    my_fft_butterfly_s4(8, y, x);
    my_fft_bottom2_butterfly(16, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_64points(int N, complex_t<T> *x)
{
    complex_t<T> y[64];
    my_fft_first_butterfly(64, 1, x, y);
    my_fft_butterfly_s4(16, y, x);
    my_fft_bottom4_butterfly(16, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_128points(int N, complex_t<T> *x)
{
    complex_t<T> y[128];
    my_fft_first_butterfly(128, 1, x, y);
    my_fft_butterfly_s4(32, y, x);
    my_fft_butterfly(8, 16, x, y);
    my_fft_bottom2_butterfly(64, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_256points(int N, complex_t<T> *x)
{
    complex_t<T> y[256];
    my_fft_first_butterfly(256, 1, x, y);
    my_fft_butterfly_s4(64, y, x);
    my_fft_butterfly(16, 16, x, y);
    my_fft_bottom4_butterfly(64, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_512points(int N, complex_t<T> *x)
{
    complex_t<T> y[512];
    my_fft_first_butterfly(512, 1, x, y);
    my_fft_butterfly_s4(128, y, x);
    my_fft_butterfly(32, 16, x, y);
    my_fft_butterfly(8, 64, y, x);
    my_fft_bottom2_butterfly(256, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_1024points(int N, complex_t<T> *x)
{
    complex_t<T> y[1024];
    my_fft_first_butterfly(1024, 1, x, y);
    my_fft_butterfly_s4(256, y, x);
    my_fft_butterfly(64, 16, x, y);
    my_fft_butterfly(16, 64, y, x);
    my_fft_bottom4_butterfly(256, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_2048points(int N, complex_t<T> *x)
{
    complex_t<T> y[2048];
    my_fft_first_butterfly(2048, 1, x, y);
    my_fft_butterfly_s4(512, y, x);
    my_fft_butterfly(128, 16, x, y);
    my_fft_butterfly(32, 64, y, x);
    my_fft_butterfly(8, 256, x, y);
    my_fft_bottom2_butterfly(1024, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_fft_4096points(int N, complex_t<T> *x)
{
    complex_t<T> y[4096];
    my_fft_first_butterfly(4096, 1, x, y);
    my_fft_butterfly_s4(1024, y, x);
    my_fft_butterfly(256, 16, x, y);
    my_fft_butterfly(64, 64, y, x);
    my_fft_butterfly(16, 256, x, y);
    my_fft_bottom4_butterfly(1024, y, x);
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
inline void my_fft_avx_whole<T>::my_ifft_first_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n1 = N/4;
    uint16_t n2 = N/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_back[index];
    complex_t<T> *w2 = w2p_back[index];
    complex_t<T> *w3 = w3p_back[index];

    for (uint16_t p = 0; p < n1; p+=2) {
        complex_t<T> *x_p = x + p;
        complex_t<T> *y_4p = y + 4*p;
        __m128 w1p_avx = _mm_loadu_ps(&(w1[p].Re));
        __m128 w2p_avx = _mm_loadu_ps(&(w2[p].Re));
        __m128 w3p_avx = _mm_loadu_ps(&(w3[p].Re));

        __m128 a = _mm_loadu_ps(&(x_p[0].Re));
        __m128 b = _mm_loadu_ps(&(x_p[n1].Re));
        __m128 c = _mm_loadu_ps(&(x_p[n2].Re));
        __m128 d = _mm_loadu_ps(&(x_p[n3].Re));
        __m128 apc = _mm_add_ps(a, c);
        __m128 amc = _mm_sub_ps(a, c);
        __m128 bpd = _mm_add_ps(b, d);
        __m128 jbmd = jxpz(_mm_sub_ps(b, d));

        __m128 aA = _mm_add_ps(apc, bpd);
        __m128 bB = mulpz_lh(w1p_avx, _mm_add_ps(amc, jbmd));
        __m128 cC = mulpz_lh(w2p_avx, _mm_sub_ps(apc, bpd));
        __m128 dD = mulpz_lh(w3p_avx, _mm_sub_ps(amc, jbmd));
        __m128 ab = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[0].Re), ab);
        __m128 cd = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(1, 0, 1, 0));
        _mm_storeu_ps(&(y_4p[2].Re), cd);
        __m128 AB = _mm_shuffle_ps(aA, bB, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[4].Re), AB);
        __m128 CD = _mm_shuffle_ps(cC, dD, _MM_SHUFFLE(3, 2, 3, 2));
        _mm_storeu_ps(&(y_4p[6].Re), CD);
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_butterfly_s4(int N, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n = N/4;
    uint16_t n1 = N;
    uint16_t n2 = 2*N;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_back[index];
    complex_t<T> *w2 = w2p_back[index];
    complex_t<T> *w3 = w3p_back[index];

    // s = 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        complex_t<T> *xq_sp = x + sp;
        complex_t<T> *yq_s4p = y + s4p;
        __m128 w1p_avx = _mm_set_ps(w1[p].Im, w1[p].Re, w1[p].Im, w1[p].Re);
        __m128 w2p_avx = _mm_set_ps(w2[p].Im, w2[p].Re, w2[p].Im, w2[p].Re);
        __m128 w3p_avx = _mm_set_ps(w3[p].Im, w3[p].Re, w3[p].Im, w3[p].Re);
        // Set the weight
        __m256 w1p = duppz2_lo(w1p_avx);
        __m256 w2p = duppz2_lo(w2p_avx);
        __m256 w3p = duppz2_lo(w3p_avx);
        // The points case : q = 1
        __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq_sp[n1].Re));
        __m256 c = _mm256_loadu_ps(&(xq_sp[n2].Re));
        __m256 d = _mm256_loadu_ps(&(xq_sp[n3].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

        _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(yq_s4p[4].Re), mulpz2(w1p, _mm256_add_ps(amc, jbmd)));
        _mm256_storeu_ps(&(yq_s4p[8].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
        _mm256_storeu_ps(&(yq_s4p[12].Re), mulpz2(w3p, _mm256_sub_ps(amc, jbmd)));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_butterfly(int N, int s, complex_t<T> *x, complex_t<T> *y)
{
    uint16_t n = N/4;
    uint16_t n1 = (N*s)/4;
    uint16_t n2 = (N*s)/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    complex_t<T> *w1 = w1p_back[index];
    complex_t<T> *w2 = w2p_back[index];
    complex_t<T> *w3 = w3p_back[index];

    // s != 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = s * p;
        uint16_t s4p = 4 * sp;
        __m128 w1p_avx = _mm_set_ps(w1[p].Im, w1[p].Re, w1[p].Im, w1[p].Re);
        __m128 w2p_avx = _mm_set_ps(w2[p].Im, w2[p].Re, w2[p].Im, w2[p].Re);
        __m128 w3p_avx = _mm_set_ps(w3[p].Im, w3[p].Re, w3[p].Im, w3[p].Re);
        // Set the weight
        __m256 w1p = duppz2_lo(w1p_avx);
        __m256 w2p = duppz2_lo(w2p_avx);
        __m256 w3p = duppz2_lo(w3p_avx);
        for (uint16_t q = 0; q < s; q+=4) {
            complex_t<T> *xq_sp = x + q + sp;
            complex_t<T> *yq_s4p = y + q + s4p;
            __m256 a = _mm256_loadu_ps(&(xq_sp[0].Re));
            __m256 b = _mm256_loadu_ps(&(xq_sp[n1].Re));
            __m256 c = _mm256_loadu_ps(&(xq_sp[n2].Re));
            __m256 d = _mm256_loadu_ps(&(xq_sp[n3].Re));
            __m256 apc = _mm256_add_ps(a, c);
            __m256 amc = _mm256_sub_ps(a, c);
            __m256 bpd = _mm256_add_ps(b, d);
            __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));

            _mm256_storeu_ps(&(yq_s4p[0].Re), _mm256_add_ps(apc, bpd));
            _mm256_storeu_ps(&(yq_s4p[s].Re), mulpz2(w1p, _mm256_add_ps(amc, jbmd)));
            _mm256_storeu_ps(&(yq_s4p[s*2].Re), mulpz2(w2p, _mm256_sub_ps(apc, bpd)));
            _mm256_storeu_ps(&(yq_s4p[s*3].Re), mulpz2(w3p, _mm256_sub_ps(amc, jbmd)));
        }
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_bottom2_butterfly(int s, complex_t<T> *x, complex_t<T> *y)
{
    for (uint16_t q = 0; q < s; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_sub_ps(a, b));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_bottom4_butterfly(int s, complex_t<T> *x, complex_t<T> *y)
{
    for (int q = 0; q < s; q+=4) {
        complex_t<T> *xq = x + q;
        complex_t<T> *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        __m256 c = _mm256_loadu_ps(&(xq[s*2].Re));
        __m256 d = _mm256_loadu_ps(&(xq[s*3].Re));
        __m256 apc = _mm256_add_ps(a, c);
        __m256 amc = _mm256_sub_ps(a, c);
        __m256 bpd = _mm256_add_ps(b, d);
        __m256 jbmd = jxpz2(_mm256_sub_ps(b, d));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(apc, bpd));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_add_ps(amc, jbmd));
        _mm256_storeu_ps(&(yq[s*2].Re), _mm256_sub_ps(apc, bpd));
        _mm256_storeu_ps(&(yq[s*3].Re), _mm256_sub_ps(amc, jbmd));
    }
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_16points(int N, complex_t<T> *x)
{
    complex_t<T> y[16];
    my_ifft_first_butterfly(16, 1, x, y);
    my_ifft_bottom4_butterfly(4, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_32points(int N, complex_t<T> *x)
{
    complex_t<T> y[32];
    my_ifft_first_butterfly(32, 1, x, y);
    my_ifft_butterfly_s4(8, y, x);
    my_ifft_bottom2_butterfly(16, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_64points(int N, complex_t<T> *x)
{
    complex_t<T> y[64];
    my_ifft_first_butterfly(64, 1, x, y);
    my_ifft_butterfly_s4(16, y, x);
    my_ifft_bottom4_butterfly(16, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_128points(int N, complex_t<T> *x)
{
    complex_t<T> y[128];
    my_ifft_first_butterfly(128, 1, x, y);
    my_ifft_butterfly_s4(32, y, x);
    my_ifft_butterfly(8, 16, x, y);
    my_ifft_bottom2_butterfly(64, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_256points(int N, complex_t<T> *x)
{
    complex_t<T> y[256];
    my_ifft_first_butterfly(256, 1, x, y);
    my_ifft_butterfly_s4(64, y, x);
    my_ifft_butterfly(16, 16, x, y);
    my_ifft_bottom4_butterfly(64, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_512points(int N, complex_t<T> *x)
{
    complex_t<T> y[512];
    my_ifft_first_butterfly(512, 1, x, y);
    my_ifft_butterfly_s4(128, y, x);
    my_ifft_butterfly(32, 16, x, y);
    my_ifft_butterfly(8, 64, y, x);
    my_ifft_bottom2_butterfly(256, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_1024points(int N, complex_t<T> *x)
{
    complex_t<T> y[1024];
    my_ifft_first_butterfly(1024, 1, x, y);
    my_ifft_butterfly_s4(256, y, x);
    my_ifft_butterfly(64, 16, x, y);
    my_ifft_butterfly(16, 64, y, x);
    my_ifft_bottom4_butterfly(256, x, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_2048points(int N, complex_t<T> *x)
{
    complex_t<T> y[2048];
    my_ifft_first_butterfly(2048, 1, x, y);
    my_ifft_butterfly_s4(512, y, x);
    my_ifft_butterfly(128, 16, x, y);
    my_ifft_butterfly(32, 64, y, x);
    my_ifft_butterfly(8, 256, x, y);
    my_ifft_bottom2_butterfly(1024, y, x);
}

template<typename T>
inline void my_fft_avx_whole<T>::my_ifft_4096points(int N, complex_t<T> *x)
{
    complex_t<T> y[4096];
    my_ifft_first_butterfly(4096, 1, x, y);
    my_ifft_butterfly_s4(1024, y, x);
    my_ifft_butterfly(256, 16, x, y);
    my_ifft_butterfly(64, 64, y, x);
    my_ifft_butterfly(16, 256, x, y);
    my_ifft_bottom4_butterfly(1024, y, x);
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
