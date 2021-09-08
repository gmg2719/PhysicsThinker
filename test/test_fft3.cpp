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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common/complex_t.h"
#include "signal/my_fft_avx4ts.hpp"
#include "my_time.h"

void test_forward(struct my_fft_avx_whole &handler)
{
    int points[84] = {6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632};

    for (int i = 0; i < 84; i++) {
        int num = points[i];
        complex_t<float> x1[num];
        complex_t<float> x2[num];
        for (int k = 0; k < num; k++)
        {
            x1[k].Re = cos(4*3.1415926*rand()/num);
            x1[k].Im = sin(2*3.1415926*rand()/num);

            x2[k].Re = x1[k].Re;
            x2[k].Im = x1[k].Im;
        }

        handler.my_fft(num, x1);
        handler.my_fft_compare(num, x2);

        bool result = true;
        for (int k = 0; k < num; k++) {
            // Single float precise decides the threshold is 1E-6
            if ((fabs(x1[k].Re-x2[k].Re)>1E-4) || (fabs(x1[k].Im-x2[k].Im)>1E-4)) {
                printf("x[%d] (%.6le %.6le) vs (%.6le %.6le)\n", k, x1[k].Re, x1[k].Im, x2[k].Re, x2[k].Im);
                result = false;
                break;
            }
        }
        if (result == false) {
            fprintf(stderr, "FFT result is error !\n");
        } else {
            printf("[%d] points %d FFT is OK !\n", i, num);
        }
    }
}

void test_backward(struct my_fft_avx_whole &handler)
{
    int points[84] = {6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632};

    for (int i = 0; i < 84; i++) {
        int num = points[i];
        complex_t<float> x1[num];
        complex_t<float> x2[num];
        for (int k = 0; k < num; k++)
        {
            x1[k].Re = cos(4*3.1415926*rand()/num);
            x1[k].Im = sin(2*3.1415926*rand()/num);

            x2[k].Re = x1[k].Re;
            x2[k].Im = x1[k].Im;
        }

        handler.my_ifft(num, x1);
        handler.my_ifft_compare(num, x2);

        bool result = true;
        for (int k = 0; k < num; k++) {
            // Single float precise decides the threshold is 1E-6
            if ((fabs(x1[k].Re-x2[k].Re)>1E-4) || (fabs(x1[k].Im-x2[k].Im)>1E-4)) {
                printf("x[%d] (%.6le %.6le) vs (%.6le %.6le)\n", k, x1[k].Re, x1[k].Im, x2[k].Re, x2[k].Im);
                result = false;
                break;
            }
        }
        if (result == false) {
            fprintf(stderr, "Inverse-FFT result is error !\n");
        } else {
            printf("[%d] points %d Inverse-FFT is OK !\n", i, num);
        }
    }
}

void test_avx()
{
    float res[8];
    float a[6] = {5.484939e-02, 5.131352e-01, -7.251574e-01, 4.458300e-01, -9.964180e-01, 9.193458e-02};
    float b[6] = {-7.251574e-01, 4.458300e-01, -9.964180e-01, 9.193458e-02, 5.484939e-02, 5.131352e-01};
    float c[6] = {-9.964180e-01, 9.193458e-02, 5.484939e-02, 5.131352e-01, -7.251574e-01, 4.458300e-01};
    __m256i idx1 = _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7);
    __m256i idx2 = _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7);
    __m256 t1 = _mm256_load_ps(a);
    __m256 t2 = _mm256_permutevar8x32_ps(t1, idx1);
    __m256 t3 = _mm256_permutevar8x32_ps(t1, idx2);
    __m256 w1 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, -8.660254e-01, -5.000000e-01, -8.660254e-01, 0.0, 0.0);
    __m256 w2 = _mm256_setr_ps(1.0, 0.0, -5.000000e-01, 8.660254e-01, 1.0, 0.0, 0.0, 0.0);
    __m256 w3 = _mm256_setr_ps(1.0, 0.0, 1.0, 0.0, -5.000000e-01, 8.660254e-01, 0.0, 0.0);
    t1 = mulpz2(t1, w1);
    t2 = mulpz2(t2, w2);
    t3 = mulpz2(t3, w3);
    t1 = _mm256_add_ps(_mm256_add_ps(t1, t2), t3);

    _mm256_store_ps(res, t1);
    for (int i = 0; i < 6; i++) {
        printf("%.6f ", res[i]);
    }  
}

int main(void)
{
    struct my_fft_avx_whole my_fft_avx_;
    int czt_points[90] = {6, 11, 12, 13, 17, 18, 19, 23, 24, 29, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632};

    my_fft_avx_.my_fft_czt_init(90, czt_points);

    test_forward(my_fft_avx_);
    test_backward(my_fft_avx_);

    return 0;
}
