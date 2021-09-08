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

#ifndef _MY_FFT_AVX_TS_HPP_
#define _MY_FFT_AVX_TS_HPP_            1

#include <cmath>
#include <utility>
#include "common/complex_t.h"
#include "my_simd.h"

#define SQRT2_CONSTANT          1.41421356237309504876378807303183294  // sqrt(2)
#define SQRT1_2_CONSTANT        0.707106781186547524400844362104849039  // 1/sqrt(2)

typedef complex_t<float>    ComplexFloat;

/////////////////////////////////////////////////
//             CZT related functions           //
/////////////////////////////////////////////////
inline unsigned int find_nearest_power2(unsigned int n)
{
    unsigned int v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

inline int is_power2(int n)
{
    return (n & (n - 1)) == 0;
}

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

struct my_fft_avx_whole
{
    // Forward FFT usage
    ComplexFloat *w1p_fwd[10];
    ComplexFloat *w2p_fwd[10];
    ComplexFloat *w3p_fwd[10];

    // Backward FFT usage
    ComplexFloat *w1p_back[10];
    ComplexFloat *w2p_back[10];
    ComplexFloat *w3p_back[10];

    uint16_t wp_points[10] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    uint16_t wp_srs_points[87] = {
        6, 9, 12, 15, 18, 24, 30, 36, 48, 51, 54, 60, 72, 84, 90, 96, 108, 120, 132, 
        144, 150, 156, 162, 168, 180, 192, 204, 216, 228, 240, 252, 264, 270, 276, 
        288, 300, 312, 324, 336, 348, 360, 384, 396, 408, 432, 450, 456, 480, 486, 
        504, 528, 540, 552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810, 
        816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152, 1200, 1248, 1296, 
        1344, 1350, 1440, 1458, 1500, 1536, 1584, 1620, 1632
    };
    uint16_t srs_points[84] = {
        6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
        120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
        228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
        360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
        552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
        816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
        1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
        1620, 1632
    };
    uint16_t srs_factors_num[84] = {
        2, 2, 2, 2, 2, 3, 2, 3, 3, 3,
        3, 3, 2, 4, 3, 3, 3, 3, 3, 5,
        3, 4, 2, 3, 4, 3, 3, 4, 3, 4,
        3, 3, 4, 3, 5, 3, 3, 4, 2, 4,
        3, 4, 5, 3, 3, 6, 4, 3, 5, 3,
        3, 4, 3, 5, 3, 3, 4, 5, 2, 4,
        5, 3, 4, 5, 3, 3, 6, 4, 3, 5,
        3, 3, 4, 3, 5, 3, 5, 4, 6, 5,
        2, 4, 5, 3
    };
    uint16_t srs_factors_list[84][6] = {
        {3, 2}, {3, 4}, {3, 6}, {3, 8}, {5, 6}, {3, 3, 4}, {3, 16}, {3, 3, 6}, {4, 3, 5}, {3, 3, 8},
        {4, 3, 7}, {6, 3, 5}, {32, 3}, {4, 3, 3, 3}, {8, 3, 5}, {4, 3, 11}, {16, 3, 3}, {6, 5, 5}, {4, 3, 13}, {2, 3, 3, 3, 3},
        {8, 3, 7}, {4, 3, 3, 5}, {64, 3}, {4, 3, 17}, {8, 3, 3, 3}, {4, 3, 19}, {3, 5, 16}, {4, 3, 3, 7}, {8, 3, 11}, {6, 3, 3, 5}, 
        {4, 3, 23}, {32, 3, 3}, {4, 3, 5, 5}, {8, 3, 13}, {4, 3, 3, 3, 3}, {16, 3, 7}, {4, 3, 29}, {8, 3, 3, 5}, {128, 3}, {4, 3, 3, 11},
        {8, 3, 17},{16, 3, 3, 3},{2, 3, 3, 5, 5},{8, 3, 19},{32, 3, 5},{2, 3, 3, 3, 3, 3},{8, 3, 3, 7},{16, 3, 11},{4, 3, 3, 3, 5},{8, 3, 23},
        {64, 3, 3},{8, 3, 5, 5},{16, 3, 13},{8, 3, 3, 3, 3},{32, 3, 7},{8, 3, 29},{16, 3, 3, 5},{2, 3, 5, 5, 5},{256, 3},{8, 3, 3, 11},
        {6, 3, 3, 3, 5},{16, 3, 17},{32, 3, 3, 3},{4, 3, 3, 5, 5},{16, 3, 19},{64, 3, 5},{4, 3, 3, 3, 3, 3},{16, 3, 3, 7},{32, 3, 11},{8, 3, 3, 3, 5},
        {16, 3, 23},{128, 3, 3},{16, 3, 5, 5},{32, 3, 13},{16, 3, 3, 3, 3},{64, 3, 7},{6, 3, 3, 5, 5}, {32, 3, 3, 5}, {6, 3, 3, 3, 3, 3},{4, 3, 5, 5, 5},
        {512, 3}, {16, 3, 3, 11}, {12, 3, 3, 3, 5},{3, 17, 32}
    };

    // CZT usage
    int czt_init_num;
    int *n1;
    int *n2_czt;
    ComplexFloat **chirp;
    ComplexFloat **ichirp;
    ComplexFloat **chirp_back;
    ComplexFloat **ichirp_back;
    ComplexFloat **czt_xp;
    // Cooley-Tukey usage
    ComplexFloat *ct_wp_fwd[87];
    ComplexFloat *ct_wp_back[87];

    // The construct interface
    my_fft_avx_whole();
    ~my_fft_avx_whole();

    // Some basic interfaces
    void my_fft_czt_init(int n, int *points);
    inline int index_of_czt(int points);
    inline int index_of_srs(int points);
    // Cooley-Tukey usage
    inline int index_of_srsWeight(int points);

    // FFT forward operations
    inline void my_fft_first_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_fft_butterfly_s4(int N, ComplexFloat *x, ComplexFloat *y);
    inline void my_fft_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_fft_bottom2_butterfly(int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_fft_bottom4_butterfly(int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_fft_8points(int N, ComplexFloat *x);
    inline void my_fft_16points(int N, ComplexFloat *x);
    inline void my_fft_32points(int N, ComplexFloat *x);
    inline void my_fft_64points(int N, ComplexFloat *x);
    inline void my_fft_128points(int N, ComplexFloat *x);
    inline void my_fft_256points(int N, ComplexFloat *x);
    inline void my_fft_512points(int N, ComplexFloat *x);
    inline void my_fft_1024points(int N, ComplexFloat *x);
    inline void my_fft_2048points(int N, ComplexFloat *x);
    inline void my_fft_4096points(int N, ComplexFloat *x);
    inline void my_fft_czt(int N, ComplexFloat *x);
    inline void fft_srs_inner(int N, ComplexFloat *x);
    inline void my_fft_cooley_tukey_no2(int N, ComplexFloat *x, uint16_t list_num, uint16_t *pq_list);
    // FFT backward operations
    inline void my_ifft_first_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_ifft_butterfly_s4(int N, ComplexFloat *x, ComplexFloat *y);
    inline void my_ifft_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_ifft_bottom2_butterfly(int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_ifft_bottom4_butterfly(int s, ComplexFloat *x, ComplexFloat *y);
    inline void my_ifft_8points(int N, ComplexFloat *x);
    inline void my_ifft_16points(int N, ComplexFloat *x);
    inline void my_ifft_32points(int N, ComplexFloat *x);
    inline void my_ifft_64points(int N, ComplexFloat *x);
    inline void my_ifft_128points(int N, ComplexFloat *x);
    inline void my_ifft_256points(int N, ComplexFloat *x);
    inline void my_ifft_512points(int N, ComplexFloat *x);
    inline void my_ifft_1024points(int N, ComplexFloat *x);
    inline void my_ifft_2048points(int N, ComplexFloat *x);
    inline void my_ifft_4096points(int N, ComplexFloat *x);
    inline void my_ifft_czt(int N, ComplexFloat *x);
    inline void ifft_srs_inner(int N, ComplexFloat *x);
    inline void my_ifft_cooley_tukey_no2(int N, ComplexFloat *x, uint16_t list_num, uint16_t *pq_list);

    // Provided wrapper interfaces
    void fft_power2_kernel(int N, ComplexFloat *x);
    void fft_not_power2_kernel(int N, ComplexFloat *x);
    void ifft_power2_kernel(int N, ComplexFloat *x);
    void ifft_not_power2_kernel(int N, ComplexFloat *x);
    void my_fft(int N, ComplexFloat *x);
    void my_fft_compare(int N, ComplexFloat *x);
    void my_ifft(int N, ComplexFloat *x);
    void my_ifft_compare(int N, ComplexFloat *x);
};


my_fft_avx_whole::my_fft_avx_whole()
{
    // Coefficents initialization
    for (uint8_t i = 0; i < 10; i++) {
        uint16_t points = wp_points[i];
        w1p_fwd[i] = new ComplexFloat[points/4];
        w2p_fwd[i] = new ComplexFloat[points/4];
        w3p_fwd[i] = new ComplexFloat[points/4];
        w1p_back[i] = new ComplexFloat[points/4];
        w2p_back[i] = new ComplexFloat[points/4];
        w3p_back[i] = new ComplexFloat[points/4];
        float theta = 2*M_PI/points;
        for (uint16_t p = 0; p < points/4; p++) {
            w1p_fwd[i][p] = ComplexFloat(cos(p*theta), -sin(p*theta));
            w2p_fwd[i][p] = w1p_fwd[i][p] * w1p_fwd[i][p];
            w3p_fwd[i][p] = w1p_fwd[i][p] * w2p_fwd[i][p];

            w1p_back[i][p] = ComplexFloat(cos(p*theta), sin(p*theta));
            w2p_back[i][p] = w1p_back[i][p] * w1p_back[i][p];
            w3p_back[i][p] = w1p_back[i][p] * w2p_back[i][p];
        }
    }

    for (uint16_t i = 0; i < 87; i++) {
        uint16_t points = wp_srs_points[i];
        ct_wp_fwd[i] = new ComplexFloat[points];
        ct_wp_back[i] = new ComplexFloat[points];
        float theta = 2 * M_PI / points;
        for (uint16_t p = 0; p < points; p++) {
            ct_wp_fwd[i][p] = ComplexFloat(cos(p*theta), -sin(p*theta));
            ct_wp_back[i][p] = ComplexFloat(cos(p*theta), sin(p*theta));
        }
    }

    n1 = NULL;
    n2_czt = NULL;
    chirp = NULL;
    ichirp = NULL;
    chirp_back = NULL;
    ichirp_back = NULL;
    czt_xp = NULL;
}

void my_fft_avx_whole::my_fft_czt_init(int n, int *points)
{
    czt_init_num = n;
    if (n <= 0) {
        n1 = NULL;
        n2_czt = NULL;
        chirp = NULL;
        ichirp = NULL;
        chirp_back = NULL;
        ichirp_back = NULL;
        czt_xp = NULL;
    } else {
        n1 = new int[n];
        n2_czt = new int[n];
        chirp = new ComplexFloat*[n];
        ichirp = new ComplexFloat*[n];
        chirp_back = new ComplexFloat*[n];
        ichirp_back = new ComplexFloat*[n];
        czt_xp = new ComplexFloat*[n];
    }

    for (int i = 0; i < n; i++)
    {
        int N = points[i];
        n1[i] = N;
        printf("%d points zct initialize OK !\n", N);
        // ZCT usage
        if (is_power2(N) == 0)
        {
            n2_czt[i] = 2 * find_nearest_power2(N);
            int N2_CZT = n2_czt[i];
            if (N2_CZT > 4096) {
                fprintf(stderr, "NOT SUPPORTED Points for ZCT transformation !\n");
                exit(-1);
            }

            chirp[i] = new ComplexFloat[2*N-1];
            ichirp[i] = new ComplexFloat[N2_CZT];
            czt_xp[i] = new ComplexFloat[N2_CZT];
            chirp_back[i] = new ComplexFloat[2*N-1];
            ichirp_back[i] = new ComplexFloat[N2_CZT];

            for (int k = 0; k < N2_CZT; k++) {
                ichirp[i][k].Re = 0.;  ichirp[i][k].Im = 0.;
                ichirp_back[i][k].Re = 0.;  ichirp_back[i][k].Im = 0.;
            }

            int index = 0;
            for (int k = 1-N; k < N; k++) {
                    float tmp = (k * k) / 2.0;
                    chirp[i][index] = ComplexFloat(cos(2*M_PI*tmp/N), -sin(2*M_PI*tmp/N));
                    chirp_back[i][index] = ComplexFloat(cos(2*M_PI*tmp/N), sin(2*M_PI*tmp/N));
                    ichirp[i][index] = 1.0 / chirp[i][index];
                    ichirp_back[i][index] = 1.0 / chirp_back[i][index];
                    index += 1;
            }

            // Get the FFT results
            my_fft(N2_CZT, ichirp[i]);
            my_fft(N2_CZT, ichirp_back[i]);
        }
        else
        {
            chirp[i] = NULL;
            ichirp[i] = NULL;
            czt_xp[i] = NULL;
            chirp_back[i] = NULL;
            ichirp_back[i] = NULL;
            fprintf(stderr, "Error some init points for ZCT in my_fft_czt_init() !\n");
        }
    }
}

inline int my_fft_avx_whole::index_of_czt(int points)
{
    for (int i = 0; i < czt_init_num; i++) {
        if (n1[i] == points) {
            return i;
        }
    }
    return -1;
}
inline int my_fft_avx_whole::index_of_srs(int points)
{
    for (int i = 0; i < 84; i++) {
        if (srs_points[i] == points) {
            return i;
        }
    }
    return -1;
}

inline int my_fft_avx_whole::index_of_srsWeight(int points)
{
    uint16_t first = 0;
    uint16_t last = 86;
    uint16_t middle;
    uint16_t search = points;

    while (first <= last) {
        middle = first + (last-first)/2;
        if (wp_srs_points[middle] < search)
        {
            first = middle + 1;
        }
        else if (wp_srs_points[middle] == search)
        {
            return middle;
        }
        else
        {
            last = middle - 1;
        }
    }
    return 86;
}

my_fft_avx_whole::~my_fft_avx_whole()
{
    for (uint8_t i = 0; i < 10; i++) {
        delete [](w1p_fwd[i]);
        delete [](w2p_fwd[i]);
        delete [](w3p_fwd[i]);
        delete [](w1p_back[i]);
        delete [](w2p_back[i]);
        delete [](w3p_back[i]);
    }

    if (n1 != NULL)
        delete []n1;
    if (n2_czt != NULL)
        delete []n2_czt;
    if (chirp != NULL)
        for (int k = 0; k < czt_init_num; k++)
            delete [](chirp[k]);
    if (ichirp != NULL)
        for (int k = 0; k < czt_init_num; k++)
            delete [](ichirp[k]);
    if (chirp_back != NULL)
        for (int k = 0; k < czt_init_num; k++)
            delete [](chirp_back[k]);
    if (ichirp_back != NULL)
        for (int k = 0; k < czt_init_num; k++)
            delete [](ichirp_back[k]);
    if (czt_xp != NULL)
        for (int k = 0; k < czt_init_num; k++)
            delete [](czt_xp[k]);
}

inline void my_fft_avx_whole::my_fft_8points(int N, ComplexFloat *x)
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

inline void my_fft_avx_whole::my_fft_first_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n1 = N/4;
    uint16_t n2 = N/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_fwd[index];
    ComplexFloat *w2 = w2p_fwd[index];
    ComplexFloat *w3 = w3p_fwd[index];

    for (uint16_t p = 0; p < n1; p+=2) {
        ComplexFloat *x_p = x + p;
        ComplexFloat *y_4p = y + 4*p;
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
        printf("p = %d %d %d %d\n", p, n1, n2, n3);
        float ans[4];
        _mm_store_ps(ans, w1p_avx);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
        _mm_store_ps(ans, w2p_avx);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
        _mm_store_ps(ans, w3p_avx);
        printf("%.8e %.8e %.8e %.8e \n", ans[0], ans[1], ans[2], ans[3]);
#endif
    }
}

inline void my_fft_avx_whole::my_fft_butterfly_s4(int N, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n = N/4;
    uint16_t n1 = N;
    uint16_t n2 = 2*N;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_fwd[index];
    ComplexFloat *w2 = w2p_fwd[index];
    ComplexFloat *w3 = w3p_fwd[index];

    // s = 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        ComplexFloat *xq_sp = x + sp;
        ComplexFloat *yq_s4p = y + s4p;
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

inline void my_fft_avx_whole::my_fft_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n = N/4;
    uint16_t n1 = (N*s)/4;
    uint16_t n2 = (N*s)/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_fwd[index];
    ComplexFloat *w2 = w2p_fwd[index];
    ComplexFloat *w3 = w3p_fwd[index];

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
            ComplexFloat *xq_sp = x + q + sp;
            ComplexFloat *yq_s4p = y + q + s4p;
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

inline void my_fft_avx_whole::my_fft_bottom2_butterfly(int s, ComplexFloat *x, ComplexFloat *y)
{
    for (uint16_t q = 0; q < s; q+=4) {
        ComplexFloat *xq = x + q;
        ComplexFloat *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_sub_ps(a, b));
    }
}

inline void my_fft_avx_whole::my_fft_bottom4_butterfly(int s, ComplexFloat *x, ComplexFloat *y)
{
    for (int q = 0; q < s; q+=4) {
        ComplexFloat *xq = x + q;
        ComplexFloat *yq = y + q;

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

inline void my_fft_avx_whole::my_fft_16points(int N, ComplexFloat *x)
{
    ComplexFloat y[16];
    my_fft_first_butterfly(16, 1, x, y);
    my_fft_bottom4_butterfly(4, y, x);
}

inline void my_fft_avx_whole::my_fft_32points(int N, ComplexFloat *x)
{
    ComplexFloat y[32];
    my_fft_first_butterfly(32, 1, x, y);
    my_fft_butterfly_s4(8, y, x);
    my_fft_bottom2_butterfly(16, x, x);
}

inline void my_fft_avx_whole::my_fft_64points(int N, ComplexFloat *x)
{
    ComplexFloat y[64];
    my_fft_first_butterfly(64, 1, x, y);
    my_fft_butterfly_s4(16, y, x);
    my_fft_bottom4_butterfly(16, x, x);
}

inline void my_fft_avx_whole::my_fft_128points(int N, ComplexFloat *x)
{
    ComplexFloat y[128];
    my_fft_first_butterfly(128, 1, x, y);
    my_fft_butterfly_s4(32, y, x);
    my_fft_butterfly(8, 16, x, y);
    my_fft_bottom2_butterfly(64, y, x);
}

inline void my_fft_avx_whole::my_fft_256points(int N, ComplexFloat *x)
{
    ComplexFloat y[256];
    my_fft_first_butterfly(256, 1, x, y);
    my_fft_butterfly_s4(64, y, x);
    my_fft_butterfly(16, 16, x, y);
    my_fft_bottom4_butterfly(64, y, x);
}

inline void my_fft_avx_whole::my_fft_512points(int N, ComplexFloat *x)
{
    ComplexFloat y[512];
    my_fft_first_butterfly(512, 1, x, y);
    my_fft_butterfly_s4(128, y, x);
    my_fft_butterfly(32, 16, x, y);
    my_fft_butterfly(8, 64, y, x);
    my_fft_bottom2_butterfly(256, x, x);
}

inline void my_fft_avx_whole::my_fft_1024points(int N, ComplexFloat *x)
{
    ComplexFloat y[1024];
    my_fft_first_butterfly(1024, 1, x, y);
    my_fft_butterfly_s4(256, y, x);
    my_fft_butterfly(64, 16, x, y);
    my_fft_butterfly(16, 64, y, x);
    my_fft_bottom4_butterfly(256, x, x);
}

inline void my_fft_avx_whole::my_fft_2048points(int N, ComplexFloat *x)
{
    ComplexFloat y[2048];
    my_fft_first_butterfly(2048, 1, x, y);
    my_fft_butterfly_s4(512, y, x);
    my_fft_butterfly(128, 16, x, y);
    my_fft_butterfly(32, 64, y, x);
    my_fft_butterfly(8, 256, x, y);
    my_fft_bottom2_butterfly(1024, y, x);
}

inline void my_fft_avx_whole::my_fft_4096points(int N, ComplexFloat *x)
{
    ComplexFloat y[4096];
    my_fft_first_butterfly(4096, 1, x, y);
    my_fft_butterfly_s4(1024, y, x);
    my_fft_butterfly(256, 16, x, y);
    my_fft_butterfly(64, 64, y, x);
    my_fft_butterfly(16, 256, x, y);
    my_fft_bottom4_butterfly(1024, y, x);
}

inline void my_fft_avx_whole::my_fft_czt(int N, ComplexFloat *x)
{
    int czt_index = index_of_czt(N);
    int N2_CZT = n2_czt[czt_index];
    ComplexFloat *pchirp = chirp[czt_index];
    ComplexFloat *pichirp = ichirp[czt_index];
    ComplexFloat *pczt_xp = czt_xp[czt_index];

    for (int k = 0; k < N2_CZT; k++) {
        pczt_xp[k].Re = 0.;  pczt_xp[k].Im = 0.;
    }

    int index = 0;
    for (int k = N - 1; k < (2*N-1); k++)
    {
        pczt_xp[index] = x[index] * pchirp[k];
        index += 1;
    }

    fft_power2_kernel(N2_CZT, pczt_xp);
    for (int k = 0; k < N2_CZT; k++) {
        pczt_xp[k] *= pichirp[k];
    }

    ifft_power2_kernel(N2_CZT, pczt_xp);

    index = 0;
    for (int k = N-1; k < (2*N-1); k++) {
        x[index] = pczt_xp[k] * pchirp[k];
        x[index] /= N2_CZT;
        index += 1;
    }
}

inline void my_fft_avx_whole::fft_srs_inner(int N, ComplexFloat *x)
{
    if (is_power2(N) == 1) {
        fft_power2_kernel(N, x);
    } else {
        if (N == 3)
        {
            static ComplexFloat tmp[4];
            __m256i idx1 = _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7);
            __m256i idx2 = _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7);
            __m256 t1 = _mm256_loadu_ps(&(x[0].Re));
            __m256 t2 = _mm256_permutevar8x32_ps(t1, idx1);
            __m256 t3 = _mm256_permutevar8x32_ps(t1, idx2);
            __m256 w1 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, -8.6602538824e-01, -5.000000e-01, -8.6602538824e-01, 0.0, 0.0);
            __m256 w2 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, 8.6602538824e-01, 1.0, 0.0, 0.0, 0.0);
            __m256 w3 = _mm256_setr_ps(1.0, 0.0, 1.0, 0.0, -5.000000e-01, 8.660254e-01, 0.0, 0.0);
            t1 = mulpz2(t1, w1);
            t2 = mulpz2(t2, w2);
            t3 = mulpz2(t3, w3);
            t1 = _mm256_add_ps(_mm256_add_ps(t1, t2), t3);

            _mm256_storeu_ps(&(tmp[0].Re), t1);
            x[0] = tmp[0]; x[1] = tmp[1]; x[2] = tmp[2];
        } else if (N == 5)
        {
            ComplexFloat w1 = ComplexFloat(cos(2*M_PI/5), -sin(2*M_PI/5));
            ComplexFloat w2 = ComplexFloat(cos(4*M_PI/5), -sin(4*M_PI/5));
            ComplexFloat w3 = ComplexFloat(cos(6*M_PI/5), -sin(6*M_PI/5));
            ComplexFloat w4 = ComplexFloat(cos(8*M_PI/5), -sin(8*M_PI/5));
            ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
            ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; 

            x[0] = tmp0 + tmp1 + tmp2 + tmp3 + tmp4;
            x[1] = tmp0 + w1 * tmp1 + w2 * tmp2 + w3 * tmp3 + w4 * tmp4;
            x[2] = tmp0 + w2 * tmp1 + w4 * tmp2 + w1 * tmp3 + w3 * tmp4;
            x[3] = tmp0 + w3 * tmp1 + w1 * tmp2 + w4 * tmp3 + w2 * tmp4;
            x[4] = tmp0 + w4 * tmp1 + w3 * tmp2 + w2 * tmp3 + w1 * tmp4;
        }
        else if (N == 6)
        {
            ComplexFloat w1 = ComplexFloat(cos(2*M_PI/6), -sin(2*M_PI/6));
            ComplexFloat w2 = ComplexFloat(cos(4*M_PI/6), -sin(4*M_PI/6));
            ComplexFloat w4 = ComplexFloat(cos(8*M_PI/6), -sin(8*M_PI/6));
            ComplexFloat w5 = ComplexFloat(cos(10*M_PI/6), -sin(10*M_PI/6));
            ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
            ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; ComplexFloat tmp5 = x[5]; 

            x[0] = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
            x[1] = tmp0 + w1 * tmp1 + w2 * tmp2 - tmp3 + w4 * tmp4 + w5 * tmp5;
            x[2] = tmp0 + w2 * tmp1 + w4 * tmp2 + tmp3 + w2 * tmp4 + w4 * tmp5;
            x[3] = tmp0 - tmp1 + tmp2 - tmp3 + tmp4 - tmp5;
            x[4] = tmp0 + w4 * tmp1 + w2 * tmp2 + tmp3 + w4 * tmp4 + w2 * tmp5;
            x[5] = tmp0 + w5 * tmp1 + w4 * tmp2 - tmp3 + w2 * tmp4 + w1 * tmp5;
        } 
        else if (N == 7) 
        {
            ComplexFloat w1 = ComplexFloat(cos(2*M_PI/7), -sin(2*M_PI/7));
            ComplexFloat w2 = ComplexFloat(cos(4*M_PI/7), -sin(4*M_PI/7));
            ComplexFloat w3 = ComplexFloat(cos(6*M_PI/7), -sin(6*M_PI/7));
            ComplexFloat w4 = ComplexFloat(cos(8*M_PI/7), -sin(8*M_PI/7));
            ComplexFloat w5 = ComplexFloat(cos(10*M_PI/7), -sin(10*M_PI/7));
            ComplexFloat w6 = ComplexFloat(cos(12*M_PI/7), -sin(12*M_PI/7));
            ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
            ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; ComplexFloat tmp5 = x[5];
            ComplexFloat tmp6 = x[6];

            x[0] = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6;
            x[1] = tmp0 + w1 * tmp1 + w2 * tmp2 + w3 * tmp3 + w4 * tmp4 + w5 * tmp5 + w6 * tmp6;
            x[2] = tmp0 + w2 * tmp1 + w4 * tmp2 + w6 * tmp3 + w1 * tmp4 + w3 * tmp5 + w5 * tmp6;
            x[3] = tmp0 + w3 * tmp1 + w6 * tmp2 + w2 * tmp3 + w5 * tmp4 + w1 * tmp5 + w4 * tmp6;
            x[4] = tmp0 + w4 * tmp1 + w1 * tmp2 + w5 * tmp3 + w2 * tmp4 + w6 * tmp5 + w3 * tmp6;
            x[5] = tmp0 + w5 * tmp1 + w3 * tmp2 + w1 * tmp3 + w6 * tmp4 + w4 * tmp5 + w2 * tmp6;
            x[6] = tmp0 + w6 * tmp1 + w5 * tmp2 + w4 * tmp3 + w3 * tmp4 + w2 * tmp5 + w1 * tmp6;
        }
        else
        {
            my_fft_czt(N, x);
        }
    }
}

inline void my_fft_avx_whole::my_fft_cooley_tukey_no2(int N, ComplexFloat *x, uint16_t list_num, uint16_t *pq_list)
{
    uint16_t pnum = pq_list[0];
    uint16_t qnum = pq_list[list_num-1];

    for (int i = 1; i < list_num-1; i++) {
        pnum *= pq_list[i];
    }

    ComplexFloat X1[qnum*pnum];

    for (int q = 0; q < qnum; q++) {
        for (int s = 0; s < pnum; s++) {
            X1[q * pnum + s] = x[s*qnum + q];
        }
    }

    if (list_num == 2) {
        for (int q = 0; q < qnum; q++) {
            fft_srs_inner(pnum, &(X1[q*pnum]));
        }
    } else {
        for (int q = 0; q < qnum; q++) {
            my_fft_cooley_tukey_no2(pnum, &(X1[q*pnum]), list_num-1, pq_list);
        }
    }

    ComplexFloat *wt = ct_wp_fwd[index_of_srsWeight(N)];
    for (int q = 0; q < qnum; q++) {
        for (int s = 0; s < pnum; s++) {
            int index = q * s;
            X1[q * pnum + s] *= wt[index];
        }
    }

    for (int s = 0; s < pnum; s++) {
        ComplexFloat tmp[qnum];
        for (int q = 0; q < qnum; q++) {
            tmp[q] = X1[q * pnum +s];
        }
        fft_srs_inner(qnum, tmp);
        for (int q = 0; q < qnum; q++) {
            x[q * pnum +s] = tmp[q];
        }
    }
}

void my_fft_avx_whole::fft_power2_kernel(int N, ComplexFloat *x)
{
    switch (N)
    {
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
            fprintf(stderr, "Not supported points of power 2 > 4096 !\n");
        } break;
    }
}

void my_fft_avx_whole::fft_not_power2_kernel(int N, ComplexFloat *x)
{
    int pos_srs = index_of_srs(N);

    if (pos_srs >= 0) {
        uint16_t num_of_list = srs_factors_num[pos_srs];
        uint16_t *p_list = srs_factors_list[pos_srs];
        my_fft_cooley_tukey_no2(N, x, num_of_list, p_list);
    } else {
        // The other situation through CZT method
        my_fft_czt(N, x);
    }
}

// Fourier transform
// N : sequence length
// x : input/output sequence
void my_fft_avx_whole::my_fft(int N, ComplexFloat *x)
{
    if (N == 1)  return;

    if (is_power2(N) == 0) {
        fft_not_power2_kernel(N, x);
    }
    else
    {
        fft_power2_kernel(N, x);
    }
}

void my_fft_avx_whole::my_fft_compare(int N, ComplexFloat *x)
{
    if (N == 1)  return;

    if (is_power2(N) == 0) {
        my_fft_czt(N, x);
    }
    else
    {
        fft_power2_kernel(N, x);
    }
}

inline void my_fft_avx_whole::my_ifft_8points(int N, ComplexFloat *x)
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

inline void my_fft_avx_whole::my_ifft_first_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n1 = N/4;
    uint16_t n2 = N/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_back[index];
    ComplexFloat *w2 = w2p_back[index];
    ComplexFloat *w3 = w3p_back[index];

    for (uint16_t p = 0; p < n1; p+=2) {
        ComplexFloat *x_p = x + p;
        ComplexFloat *y_4p = y + 4*p;
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

inline void my_fft_avx_whole::my_ifft_butterfly_s4(int N, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n = N/4;
    uint16_t n1 = N;
    uint16_t n2 = 2*N;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_back[index];
    ComplexFloat *w2 = w2p_back[index];
    ComplexFloat *w3 = w3p_back[index];

    // s = 4
    for (uint16_t p = 0; p < n; p++) {
        uint16_t sp = 4 * p;
        uint16_t s4p = 4 * sp;
        ComplexFloat *xq_sp = x + sp;
        ComplexFloat *yq_s4p = y + s4p;
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

inline void my_fft_avx_whole::my_ifft_butterfly(int N, int s, ComplexFloat *x, ComplexFloat *y)
{
    uint16_t n = N/4;
    uint16_t n1 = (N*s)/4;
    uint16_t n2 = (N*s)/2;
    uint16_t n3 = n1 + n2;
    uint8_t index = my_position_of_weight(N);
    ComplexFloat *w1 = w1p_back[index];
    ComplexFloat *w2 = w2p_back[index];
    ComplexFloat *w3 = w3p_back[index];

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
            ComplexFloat *xq_sp = x + q + sp;
            ComplexFloat *yq_s4p = y + q + s4p;
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

inline void my_fft_avx_whole::my_ifft_bottom2_butterfly(int s, ComplexFloat *x, ComplexFloat *y)
{
    for (uint16_t q = 0; q < s; q+=4) {
        ComplexFloat *xq = x + q;
        ComplexFloat *yq = y + q;

        __m256 a = _mm256_loadu_ps(&(xq[0].Re));
        __m256 b = _mm256_loadu_ps(&(xq[s].Re));
        _mm256_storeu_ps(&(yq[0].Re), _mm256_add_ps(a, b));
        _mm256_storeu_ps(&(yq[s].Re), _mm256_sub_ps(a, b));
    }
}

inline void my_fft_avx_whole::my_ifft_bottom4_butterfly(int s, ComplexFloat *x, ComplexFloat *y)
{
    for (int q = 0; q < s; q+=4) {
        ComplexFloat *xq = x + q;
        ComplexFloat *yq = y + q;

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

inline void my_fft_avx_whole::my_ifft_16points(int N, ComplexFloat *x)
{
    ComplexFloat y[16];
    my_ifft_first_butterfly(16, 1, x, y);
    my_ifft_bottom4_butterfly(4, y, x);
}

inline void my_fft_avx_whole::my_ifft_32points(int N, ComplexFloat *x)
{
    ComplexFloat y[32];
    my_ifft_first_butterfly(32, 1, x, y);
    my_ifft_butterfly_s4(8, y, x);
    my_ifft_bottom2_butterfly(16, x, x);
}

inline void my_fft_avx_whole::my_ifft_64points(int N, ComplexFloat *x)
{
    ComplexFloat y[64];
    my_ifft_first_butterfly(64, 1, x, y);
    my_ifft_butterfly_s4(16, y, x);
    my_ifft_bottom4_butterfly(16, x, x);
}

inline void my_fft_avx_whole::my_ifft_128points(int N, ComplexFloat *x)
{
    ComplexFloat y[128];
    my_ifft_first_butterfly(128, 1, x, y);
    my_ifft_butterfly_s4(32, y, x);
    my_ifft_butterfly(8, 16, x, y);
    my_ifft_bottom2_butterfly(64, y, x);
}

inline void my_fft_avx_whole::my_ifft_256points(int N, ComplexFloat *x)
{
    ComplexFloat y[256];
    my_ifft_first_butterfly(256, 1, x, y);
    my_ifft_butterfly_s4(64, y, x);
    my_ifft_butterfly(16, 16, x, y);
    my_ifft_bottom4_butterfly(64, y, x);
}

inline void my_fft_avx_whole::my_ifft_512points(int N, ComplexFloat *x)
{
    ComplexFloat y[512];
    my_ifft_first_butterfly(512, 1, x, y);
    my_ifft_butterfly_s4(128, y, x);
    my_ifft_butterfly(32, 16, x, y);
    my_ifft_butterfly(8, 64, y, x);
    my_ifft_bottom2_butterfly(256, x, x);
}

inline void my_fft_avx_whole::my_ifft_1024points(int N, ComplexFloat *x)
{
    ComplexFloat y[1024];
    my_ifft_first_butterfly(1024, 1, x, y);
    my_ifft_butterfly_s4(256, y, x);
    my_ifft_butterfly(64, 16, x, y);
    my_ifft_butterfly(16, 64, y, x);
    my_ifft_bottom4_butterfly(256, x, x);
}

inline void my_fft_avx_whole::my_ifft_2048points(int N, ComplexFloat *x)
{
    ComplexFloat y[2048];
    my_ifft_first_butterfly(2048, 1, x, y);
    my_ifft_butterfly_s4(512, y, x);
    my_ifft_butterfly(128, 16, x, y);
    my_ifft_butterfly(32, 64, y, x);
    my_ifft_butterfly(8, 256, x, y);
    my_ifft_bottom2_butterfly(1024, y, x);
}

inline void my_fft_avx_whole::my_ifft_4096points(int N, ComplexFloat *x)
{
    ComplexFloat y[4096];
    my_ifft_first_butterfly(4096, 1, x, y);
    my_ifft_butterfly_s4(1024, y, x);
    my_ifft_butterfly(256, 16, x, y);
    my_ifft_butterfly(64, 64, y, x);
    my_ifft_butterfly(16, 256, x, y);
    my_ifft_bottom4_butterfly(1024, y, x);
}

inline void my_fft_avx_whole::my_ifft_czt(int N, ComplexFloat *x)
{
    int czt_index = index_of_czt(N);
    int N2_CZT = n2_czt[czt_index];
    ComplexFloat *pchirp = chirp_back[czt_index];
    ComplexFloat *pichirp = ichirp_back[czt_index];
    ComplexFloat *pczt_xp = czt_xp[czt_index];

    for (int k = 0; k < N2_CZT; k++) {
        pczt_xp[k].Re = 0.;  pczt_xp[k].Im = 0.;
    }

    int index = 0;
    for (int k = N - 1; k < (2*N-1); k++)
    {
        pczt_xp[index] = x[index] * pchirp[k];
        index += 1;
    }

    fft_power2_kernel(N2_CZT, pczt_xp);
    for (int k = 0; k < N2_CZT; k++) {
        pczt_xp[k] *= pichirp[k];
    }

    ifft_power2_kernel(N2_CZT, pczt_xp);

    index = 0;
    for (int k = N-1; k < (2*N-1); k++) {
        x[index] = pczt_xp[k] * pchirp[k];
        x[index] /= (N2_CZT*N);
        index += 1;
    }
}

inline void my_fft_avx_whole::ifft_srs_inner(int N, ComplexFloat *x)
{
    if (is_power2(N) == 1) {
        ifft_power2_kernel(N, x);

        for (int i = 0; i < N; i++)
        {
            x[i] /= N;
        }
    } else {
    switch (N)
    {
        case 3 : {
                static ComplexFloat tmp[4];
                __m256i idx1 = _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7);
                __m256i idx2 = _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7);
                __m256 t1 = _mm256_loadu_ps(&(x[0].Re));
                __m256 t2 = _mm256_permutevar8x32_ps(t1, idx1);
                __m256 t3 = _mm256_permutevar8x32_ps(t1, idx2);
                __m256 w1 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, 8.6602538824e-01, -5.000000e-01, 8.6602538824e-01, 0.0, 0.0);
                __m256 w2 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, -8.6602538824e-01, 1.0, 0.0, 0.0, 0.0);
                __m256 w3 = _mm256_setr_ps(1.0, 0.0, 1.0, 0.0, -5.000000e-01, -8.660254e-01, 0.0, 0.0);
                __m256 normal_coe = _mm256_set1_ps(1.0/3);
                t1 = mulpz2(t1, w1);
                t2 = mulpz2(t2, w2);
                t3 = mulpz2(t3, w3);
                t1 = _mm256_add_ps(_mm256_add_ps(t1, t2), t3);
                t1 = _mm256_mul_ps(t1, normal_coe);

                _mm256_storeu_ps(&(tmp[0].Re), t1);
                x[0] = tmp[0]; x[1] = tmp[1]; x[2] = tmp[2];
            } break;
        case 5 : {
                ComplexFloat w1 = ComplexFloat(cos(2*M_PI/5), sin(2*M_PI/5));
                ComplexFloat w2 = ComplexFloat(cos(4*M_PI/5), sin(4*M_PI/5));
                ComplexFloat w3 = ComplexFloat(cos(6*M_PI/5), sin(6*M_PI/5));
                ComplexFloat w4 = ComplexFloat(cos(8*M_PI/5), sin(8*M_PI/5));
                ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
                ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; 

                x[0] = (tmp0 + tmp1 + tmp2 + tmp3 + tmp4) / 5.0;
                x[1] = (tmp0 + w1 * tmp1 + w2 * tmp2 + w3 * tmp3 + w4 * tmp4) / 5.0;
                x[2] = (tmp0 + w2 * tmp1 + w4 * tmp2 + w1 * tmp3 + w3 * tmp4) / 5.0;
                x[3] = (tmp0 + w3 * tmp1 + w1 * tmp2 + w4 * tmp3 + w2 * tmp4) / 5.0;
                x[4] = (tmp0 + w4 * tmp1 + w3 * tmp2 + w2 * tmp3 + w1 * tmp4) / 5.0;
            } break;
        case 6 : {
                ComplexFloat w1 = ComplexFloat(cos(2*M_PI/6), sin(2*M_PI/6));
                ComplexFloat w2 = ComplexFloat(cos(4*M_PI/6), sin(4*M_PI/6));
                ComplexFloat w4 = ComplexFloat(cos(8*M_PI/6), sin(8*M_PI/6));
                ComplexFloat w5 = ComplexFloat(cos(10*M_PI/6), sin(10*M_PI/6));
                ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
                ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; ComplexFloat tmp5 = x[5]; 

                x[0] = (tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5) / 6.0;
                x[1] = (tmp0 + w1 * tmp1 + w2 * tmp2 - tmp3 + w4 * tmp4 + w5 * tmp5) / 6.0;
                x[2] = (tmp0 + w2 * tmp1 + w4 * tmp2 + tmp3 + w2 * tmp4 + w4 * tmp5) / 6.0;
                x[3] = (tmp0 - tmp1 + tmp2 - tmp3 + tmp4 - tmp5) / 6.0;
                x[4] = (tmp0 + w4 * tmp1 + w2 * tmp2 + tmp3 + w4 * tmp4 + w2 * tmp5) / 6.0;
                x[5] = (tmp0 + w5 * tmp1 + w4 * tmp2 - tmp3 + w2 * tmp4 + w1 * tmp5) / 6.0;
            } break;
        case 7 : {
                ComplexFloat w1 = ComplexFloat(cos(2*M_PI/7), sin(2*M_PI/7));
                ComplexFloat w2 = ComplexFloat(cos(4*M_PI/7), sin(4*M_PI/7));
                ComplexFloat w3 = ComplexFloat(cos(6*M_PI/7), sin(6*M_PI/7));
                ComplexFloat w4 = ComplexFloat(cos(8*M_PI/7), sin(8*M_PI/7));
                ComplexFloat w5 = ComplexFloat(cos(10*M_PI/7), sin(10*M_PI/7));
                ComplexFloat w6 = ComplexFloat(cos(12*M_PI/7), sin(12*M_PI/7));
                ComplexFloat tmp0 = x[0]; ComplexFloat tmp1 = x[1]; ComplexFloat tmp2 = x[2];
                ComplexFloat tmp3 = x[3]; ComplexFloat tmp4 = x[4]; ComplexFloat tmp5 = x[5];
                ComplexFloat tmp6 = x[6];

                x[0] = (tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6) / 7.0;
                x[1] = (tmp0 + w1 * tmp1 + w2 * tmp2 + w3 * tmp3 + w4 * tmp4 + w5 * tmp5 + w6 * tmp6) / 7.0;
                x[2] = (tmp0 + w2 * tmp1 + w4 * tmp2 + w6 * tmp3 + w1 * tmp4 + w3 * tmp5 + w5 * tmp6) / 7.0;
                x[3] = (tmp0 + w3 * tmp1 + w6 * tmp2 + w2 * tmp3 + w5 * tmp4 + w1 * tmp5 + w4 * tmp6) / 7.0;
                x[4] = (tmp0 + w4 * tmp1 + w1 * tmp2 + w5 * tmp3 + w2 * tmp4 + w6 * tmp5 + w3 * tmp6) / 7.0;
                x[5] = (tmp0 + w5 * tmp1 + w3 * tmp2 + w1 * tmp3 + w6 * tmp4 + w4 * tmp5 + w2 * tmp6) / 7.0;
                x[6] = (tmp0 + w6 * tmp1 + w5 * tmp2 + w4 * tmp3 + w3 * tmp4 + w2 * tmp5 + w1 * tmp6) / 7.0;
            } break;
        default : {
                // The other situation
                my_ifft_czt(N, x);
            } break;
        }
    }
}

inline void my_fft_avx_whole::my_ifft_cooley_tukey_no2(int N, ComplexFloat *x, uint16_t list_num, uint16_t *pq_list)
{
    uint16_t pnum = pq_list[0];
    uint16_t qnum = pq_list[list_num-1];

    for (int i = 1; i < list_num-1; i++) {
        pnum *= pq_list[i];
    }

    ComplexFloat X1[qnum*pnum];

    for (int q = 0; q < qnum; q++) {
        for (int s = 0; s < pnum; s++) {
            X1[q * pnum + s] = x[s*qnum + q];
        }
    }

    if (list_num == 2) {
        for (int q = 0; q < qnum; q++) {
            ifft_srs_inner(pnum, &(X1[q*pnum]));
        }
    } else {
        for (int q = 0; q < qnum; q++) {
            my_ifft_cooley_tukey_no2(pnum, &(X1[q*pnum]), list_num-1, pq_list);
        }
    }

    ComplexFloat *wt = ct_wp_back[index_of_srsWeight(N)];
    for (int q = 0; q < qnum; q++) {
        for (int s = 0; s < pnum; s++) {
            int index = q * s;
            X1[q * pnum + s] *= wt[index];
        }
    }

    for (int s = 0; s < pnum; s++) {
        ComplexFloat tmp[qnum];
        for (int q = 0; q < qnum; q++) {
            tmp[q] = X1[q * pnum +s];
        }
        ifft_srs_inner(qnum, tmp);
        for (int q = 0; q < qnum; q++) {
            x[q * pnum +s] = tmp[q];
        }
    }
}

void my_fft_avx_whole::ifft_power2_kernel(int N, ComplexFloat *x)
{
    switch (N)
    {
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
            fprintf(stderr, "Not supported points of power 2 > 4096 !\n");
        } break;
    }
}

void my_fft_avx_whole::ifft_not_power2_kernel(int N, ComplexFloat *x)
{
    int pos_srs = index_of_srs(N);

    if (pos_srs >= 0) {
        uint16_t num_of_list = srs_factors_num[pos_srs];
        uint16_t *p_list = srs_factors_list[pos_srs];
        my_ifft_cooley_tukey_no2(N, x, num_of_list, p_list);
    } else {
        // The other situation through CZT method
        my_ifft_czt(N, x);
    }
}

// Inverse Fourier transform
// N : sequence length
// x : input/output sequence
void my_fft_avx_whole::my_ifft(int N, ComplexFloat *x)
{
    if (N == 1)  return;

    if (is_power2(N) == 0) {
        ifft_not_power2_kernel(N, x);
    }
    else
    {
        ifft_power2_kernel(N, x);
    }
}

void my_fft_avx_whole::my_ifft_compare(int N, ComplexFloat *x)
{
    if (N == 1)  return;

    if (is_power2(N) == 0) {
        my_ifft_czt(N, x);
    }
    else
    {
        ifft_power2_kernel(N, x);
    }
}

#endif
