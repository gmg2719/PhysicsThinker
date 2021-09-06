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

void test1()
{
    struct my_fft_avx_whole my_fft_avx_;
    int points[84] = {6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632};
    my_fft_avx_.my_fft_czt_init(84, points);

    complex_t<float> x[24];

    for (int i = 0; i < 10; i++) {
        printf("RUN TIME = %d\n", i);
        x[0].Re = 7.62435794e-01;   x[0].Im = 2.06383348e-01;
        x[1].Re = 1.95438373e+00;   x[1].Im = -1.29716802e+00;
        x[2].Re = -3.51395518e-01;  x[2].Im = 2.51173091e+00;
        x[3].Re = 8.30021858e-01;   x[3].Im = 2.47798109e+00;
        x[4].Re = -8.85782421e-01;  x[4].Im = 1.04149783e+00;
        x[5].Re = -1.41291881e+00;  x[5].Im = 2.89411402e+00;
        x[6].Re = -1.00015211e+00;  x[6].Im = -1.37304044e+00;
        x[7].Re = -2.28566742e+00;  x[7].Im = -6.59287274e-01;
        x[8].Re = 1.04745364e+00;   x[8].Im = 7.48452485e-01;
        x[9].Re = 1.25504541e+00;   x[9].Im = -4.69390452e-01;
        x[10].Re = -4.25973117e-01; x[10].Im = 1.34006751e+00;
        x[11].Re = 1.77294597e-01;  x[11].Im = 8.03263605e-01;
        x[12].Re = -1.19099844e+00; x[12].Im = 3.62012446e-01;
        x[13].Re = -1.95291626e+00; x[13].Im = 1.21275023e-01;
        x[14].Re = 1.28068149e+00;  x[14].Im = -2.16396064e-01;
        x[15].Re = -9.94455218e-01; x[15].Im = -1.08508790e+00;
        x[16].Re = 1.63691080e+00;  x[16].Im = 1.24296121e-01;
        x[17].Re = 1.35439610e+00;  x[17].Im = - 2.50292659e+00;
        x[18].Re = 4.71289456e-02;  x[18].Im = 1.99719679e+00;
        x[19].Re = 2.34237742e+00;  x[19].Im = 1.72555804e+00;
        x[20].Re = -1.30372810e+00; x[20].Im = 3.60458732e-01;
        x[21].Re = -1.52314532e+00; x[21].Im = 1.17943203e+00;
        x[22].Re = -6.24070354e-02; x[22].Im = -1.74195826e+00;
        x[23].Re = -9.84873921e-02; x[23].Im = -1.50130713e+00;
        my_fft_avx_.my_fft(24, x);

        for (int k = 0; k < 24; k++) {
            printf("%.8e, %.8e\n", x[k].Re, x[k].Im);
        }
    }
}

int main(void)
{
    struct my_fft_avx_whole my_fft_avx_;
    int points[84] = {6, 12, 18, 24, 30, 36, 48, 54, 60, 72, 84, 90, 96, 108,
                    120, 132, 144, 150, 156, 162, 168, 180, 192, 204, 216,
                    228, 240, 252, 264, 270, 276, 288, 300, 312, 324, 336, 348,
                    360, 384, 396, 408, 432, 450, 456, 480, 486, 504, 528, 540,
                    552, 576, 600, 624, 648, 672, 696, 720, 750, 768, 792, 810,
                    816, 864, 900, 912, 960, 972, 1008, 1056, 1080, 1104, 1152,
                    1200, 1248, 1296, 1344, 1350, 1440, 1458, 1500, 1536, 1584,
                    1620, 1632};
    my_fft_avx_.my_fft_czt_init(84, points);

    complex_t<float> x[3];
    x[0].Re = 0.77132064; x[0].Im = 0.74880388;
    x[1].Re = 0.02075195; x[1].Im = 0.49850701;
    x[2].Re = 0.63364823; x[2].Im = 0.22479665;
    my_fft_avx_.my_ifft(3, x);

    for (int k = 0; k < 3; k++) {
        printf("%.8e, %.8e\n", x[k].Re, x[k].Im);
    }

    return 0;
}
