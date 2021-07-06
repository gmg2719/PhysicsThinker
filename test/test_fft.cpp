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
#include "common/complex_t.h1"
#include "signal/fft_cooley_tukey.hpp"
#include "signal/fft_stockham.hpp"

#define RUNNING_STATICS_COUNT       10

enum fft_method_type {
    COOLEY_TUKEY_C = 1,
    STOCKHAM_C = 2,
    ITERATIVE_STOCKHAM_C = 3
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
        fft_cooley_tukey<float>(N, x[i]);
    }
    end = my_us_gettimeofday();

    ans = float(end - start);

    for (int i = 0; i < 10000; i++) {
        delete [](x[i]);
    }

    return ans;
}

void run_fft_benchmark_app1()
{
    int points[12] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    float run_time_cooley_tukey[12] = {0.};
    float run_time_stockham[12] = {0.};

    for (int i = 0; i < 12; i++)
    {
        int fft_points = points[i];
        for (int running = 0; running < RUNNING_STATICS_COUNT; running++)
        {
            run_time_cooley_tukey[i] += fft_fwd_testor(COOLEY_TUKEY_C, fft_points);
            run_time_stockham[i] += fft_fwd_testor(STOCKHAM_C, fft_points);
        }

        run_time_cooley_tukey[i] /= RUNNING_STATICS_COUNT;
        run_time_stockham[i] /= RUNNING_STATICS_COUNT;
    }

    printf("Final results (us) :\n");
    printf("Points are (2 4 6 8 16 32 64 128 256 512 1024 2048 4096)\n");
    printf_runtime_one_line("C_cooley tukey", 12, run_time_cooley_tukey);
    printf_runtime_one_line("C_stockham", 12, run_time_stockham);
}

void fft_fwd_back_check(int type, int N)
{
}

void run_fft_benchmark_app2()
{
    int points[13] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int i = 0; i < 13; i++)
    {
        int n = points[i];
        printf("Calibration of %d : \n", n);
        fft_fwd_back_check(n);
    }
}

int main(void)
{
    run_fft_benchmark_app2();
    run_fft_benchmark_app1();

    return 0;
}
