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
#include "signal/fft_cooley_tukey.hpp"
#include "signal/fft_stockham.hpp"
#include "signal/fft_iterative_stockham.hpp"
#include "signal/fft_stockham_r4.hpp"
#include "my_time.h"

#define RUNNING_STATICS_COUNT       4

enum fft_method_type {
    COOLEY_TUKEY_C = 1,
    STOCKHAM_C = 2,
    ITERATIVE_STOCKHAM_C = 3,
    STOCKHAM_R4_C = 4
};

typedef void (*FFT_FWD_FUNC)(int N, complex_t<float> *x);

static void print_runtime_one_line(const char *name, int n, float run_time[])
{
    printf("%10s :", name);
    for (int i = 0; i < 12; i++)
    {
        printf(" %.4lf", run_time[i]);
    }
    printf("\n");
}

float fft_fwd_testor(int type, int N)
{
    float ans = 0.;
    uint64_t start, end;
    FFT_FWD_FUNC fwd;
    complex_t<float> *x[10000];

    if (type == COOLEY_TUKEY_C) {
        fwd = fft_cooley_tukey<float>;
    } else if (type == STOCKHAM_C) {
        fwd = fft_stockham<float>;
    } else if (type == ITERATIVE_STOCKHAM_C) {
        fwd = fft_stockham_iterative<float>;
    } else if (type == STOCKHAM_R4_C) {
        fwd = fft_stockham_r4<float>;
    } else {
        fwd = fft_cooley_tukey<float>;
    }

    for (int i = 0; i < 10000; i++) {
        x[i] = new complex_t<float>[N];
        for (int k = 0; k < N; k++)
        {
            x[i][k].Re = cos(6*3.1415926*k/N);
            x[i][k].Im = 0;
        }
    }

    start = my_us_gettimeofday();
    for (int i = 0; i < 10000; i++) {
        fwd(N, x[i]);
    }
    end = my_us_gettimeofday();

    ans = float(end - start) / 10000;

    for (int i = 0; i < 10000; i++) {
        delete [](x[i]);
    }

    return ans;
}

void run_fft_benchmark_app1()
{
    int points[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    float run_time_cooley_tukey[12] = {0.};
    float run_time_stockham[12] = {0.};
    float run_time_iterstk[12] = {0.};
    float run_time_r4[12] = {0.};

    for (int i = 0; i < 12; i++)
    {
        int fft_points = points[i];
        for (int running = 0; running < RUNNING_STATICS_COUNT; running++)
        {
            run_time_cooley_tukey[i] += fft_fwd_testor(COOLEY_TUKEY_C, fft_points);
            run_time_stockham[i] += fft_fwd_testor(STOCKHAM_C, fft_points);
            run_time_iterstk[i] += fft_fwd_testor(ITERATIVE_STOCKHAM_C, fft_points);
            run_time_r4[i] += fft_fwd_testor(STOCKHAM_R4_C, fft_points);
            printf("points(%d) running %d(%d) done !\n", fft_points, running, RUNNING_STATICS_COUNT);
        }

        run_time_cooley_tukey[i] /= RUNNING_STATICS_COUNT;
        run_time_stockham[i] /= RUNNING_STATICS_COUNT;
        run_time_iterstk[i] /= RUNNING_STATICS_COUNT;
        run_time_r4[i] /= RUNNING_STATICS_COUNT;
    }

    printf("Final results (us) :\n");
    printf("Points are (2 4 6 8 16 32 64 128 256 512 1024 2048 4096)\n");
    print_runtime_one_line("C_cooley tukey", 12, run_time_cooley_tukey);
    print_runtime_one_line("C_stockham", 12, run_time_stockham);
    print_runtime_one_line("C_iterative_stockham", 12, run_time_iterstk);
    print_runtime_one_line("C_Radix4_stockham", 12, run_time_r4);
}

// TODO : FFT fwd and backward calibration
void fft_fwd_back_check(int type, int N)
{
    FFT_FWD_FUNC fwd;
    FFT_FWD_FUNC bwd;
    complex_t<float> *x;
    complex_t<float> *x_original;

    if (type == COOLEY_TUKEY_C) {
        fwd = fft_cooley_tukey<float>;
        bwd = ifft_cooley_tukey<float>;
    } else if (type == STOCKHAM_C) {
        fwd = fft_stockham<float>;
        bwd = ifft_stockham<float>;
    } else if (type == ITERATIVE_STOCKHAM_C) {
        fwd = fft_stockham_iterative<float>;
        bwd = ifft_stockham_iterative<float>;
    } else if (type == STOCKHAM_R4_C) {
        fwd = fft_stockham_r4<float>;
        bwd = ifft_stockham_r4<float>;
    } else {
        fwd = fft_cooley_tukey<float>;
        bwd = ifft_cooley_tukey<float>;
    }

    x = new complex_t<float>[N];
    x_original = new complex_t<float>[N];
    for (int k = 0; k < N; k++)
    {
        x[k].Re = cos(4*3.1415926*rand()/N);
        x[k].Im = sin(2*3.1415926*rand()/N);
        x_original[k].Re = x[k].Re;  x_original[k].Im = x[k].Im;
    }

    fwd(N, x);
    int same_element = 0;
    for (int k = 0; k < N; k++) {
        if ((x[k].Re == x_original[k].Re) && (x[k].Im == x_original[k].Im)) {
            same_element++;
        }
    }
    if (same_element == N) {
        fprintf(stderr, "FFT result is the same as the input, maybe error !\n");
    }
    bwd(N, x);
    bool result = true;
    for (int k = 0; k < N; k++) {
        // Single float precise decides the threshold is 1E-6
        if ((fabs(x[k].Re-x_original[k].Re)>1E-6) || (fabs(x[k].Im-x_original[k].Im)>1E-6)) {
            printf("x[%d] (%.6le %.6le) vs (%.6le %.6le)\n", k, x[k].Re, x[k].Im, x_original[k].Re, x_original[k].Im);
            result = false;
            break;
        }
    }
    if (result == false) {
        fprintf(stderr, "IFFT result is error !\n");
    }

    delete []x_original;
    delete []x;
}

// FFT benchmark application2, calibration
void run_fft_benchmark_app2()
{
    int points[12] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    printf("Calibration of cooley-tukey algorithm : \n");
    for (int i = 0; i < 12; i++)
    {
        int n = points[i];
        printf("%d\n", n);
        fft_fwd_back_check(COOLEY_TUKEY_C, n);
    }

    printf("Calibration of stockham algorithm : \n");
    for (int i = 0; i < 12; i++)
    {
        int n = points[i];
        printf("%d\n", n);
        fft_fwd_back_check(STOCKHAM_C, n);
    }

    printf("Calibration of iterative-stockham algorithm : \n");
    for (int i = 0; i < 12; i++)
    {
        int n = points[i];
        printf("%d\n", n);
        fft_fwd_back_check(ITERATIVE_STOCKHAM_C, n);
    }

    printf("Calibration of Radix4-stockham algorithm : \n");
    for (int i = 0; i < 12; i++)
    {
        int n = points[i];
        printf("%d\n", n);
        fft_fwd_back_check(STOCKHAM_R4_C, n);
    }
}

int main(void)
{
    run_fft_benchmark_app2();
    run_fft_benchmark_app1();

    return 0;
}
