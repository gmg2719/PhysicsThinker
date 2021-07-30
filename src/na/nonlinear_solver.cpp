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

#include <cmath>
#include <algorithm>
#include "na/nonlinear_solver.h"

/////////////////////////////////////
//                                 //
//        Iterative methods        //
//                                 //
/////////////////////////////////////

void na_nonlinear_solver_fixedpoint(NL_Func_double func, int16_t n, double *x)
{
    if ((n <= 0) || (x == NULL) || (func == NULL)) {
        return;
    }

    for (int16_t i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int8_t conv;
    int16_t itr = 0;
    int16_t total_iters = MAX_ITERS_NLSOLVER;
    double *x_before = new double[n];

    while (itr < total_iters)
    {
        itr += 1;
        for (int16_t i = 0; i < n; i++) {
            x_before[i] = x[i];
        }

        // Calculation x = F(x) of the non-linear system
        func(n, x);

        conv = 1;
        for (int16_t i = 0; i < n; i++) {
            if (fabs(x_before[i] - x[i]) >= 1E-6)
            {
                conv = 0;
            }
        }

        if (conv == 1) {
            break;
        }
#ifdef DEBUG_PRINT_ITERS
        printf("Itr %d : (%.6f, ...)\n", itr, x[0]);
#endif
    }
    printf("Itr %d : fixedpoint method is convergence !\n", itr);

    delete []x_before;
}

void na_nonlinear_solver_fixedpoint(NL_Func_float func, int16_t n, float *x)
{
    if ((n <= 0) || (x == NULL) || (func == NULL)) {
        return;
    }

    for (int16_t i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int8_t conv;
    int16_t itr = 0;
    int16_t total_iters = MAX_ITERS_NLSOLVER;
    float *x_before = new float[n];

    while (itr < total_iters)
    {
        itr += 1;
        for (int16_t i = 0; i < n; i++) {
            x_before[i] = x[i];
        }

        // Calculation x = F(x) of the non-linear system
        func(n, x);

        conv = 1;
        for (int16_t i = 0; i < n; i++) {
            if (fabs(x_before[i] - x[i]) >= 1E-6)
            {
                conv = 0;
            }
        }

        if (conv == 1) {
            break;
        }
#ifdef DEBUG_PRINT_ITERS
        printf("Itr %d : (%.6f, ...)\n", itr, x[0]);
#endif
    }
    printf("Itr %d : fixedpoint method is convergence !\n", itr);

    delete []x_before;
}

void na_nonlinear_solver_newton(NL_DeriveFunc_double func, int16_t n, double *x)
{
    if ((n <= 0) || (x == NULL) || (func == NULL)) {
        return;
    }

    for (int16_t i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int8_t conv;
    int16_t itr = 0;
    int16_t total_iters = MAX_ITERS_NLSOLVER;
    double *delta_x = new double[n];

    while (itr < total_iters)
    {
        itr += 1;

        // Calculation dx = D(x) = G'(x) = (F(x) - x)' of the non-linear system
        for (int16_t i = 0; i < n; i++) {
            delta_x[i] = x[i];
        }
        func(n, delta_x);

        conv = 1;
        for (int16_t i = 0; i < n; i++) {
            if (fabs(delta_x[i]) >= 1E-6)
            {
                conv = 0;
            }
        }

        if (conv == 1) {
            for (int16_t i = 0; i < n; i++) {
                x[i] += delta_x[i];
            }
            break;
        }

        for (int16_t i = 0; i < n; i++) {
            x[i] += delta_x[i];
        }
#ifdef DEBUG_PRINT_ITERS
        printf("Itr %d : (%.6f, ...) ---> (%.6f, ...)\n", itr, delta_x[0], x[0]);
#endif
    }
    printf("Itr %d : newton method is convergence !\n", itr);

    delete []delta_x;
}

void na_nonlinear_solver_newton(NL_DeriveFunc_float func, int16_t n, float *x)
{
    if ((n <= 0) || (x == NULL) || (func == NULL)) {
        return;
    }

    for (int16_t i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int8_t conv;
    int16_t itr = 0;
    int16_t total_iters = MAX_ITERS_NLSOLVER;
    float *delta_x = new float[n];

    while (itr < total_iters)
    {
        itr += 1;

        // Calculation dx = D(x) = G'(x) = (F(x) - x)' of the non-linear system
        for (int16_t i = 0; i < n; i++) {
            delta_x[i] = x[i];
        }
        func(n, delta_x);

        conv = 1;
        for (int16_t i = 0; i < n; i++) {
            if (fabs(delta_x[i]) >= 1E-6)
            {
                conv = 0;
            }
        }

        if (conv == 1) {
            for (int16_t i = 0; i < n; i++) {
                x[i] += delta_x[i];
            }
            break;
        }

        for (int16_t i = 0; i < n; i++) {
            x[i] += delta_x[i];
        }
#ifdef DEBUG_PRINT_ITERS
        printf("Itr %d : (%.6f, ...) ---> (%.6f, ...)\n", itr, delta_x[0], x[0]);
#endif
    }
    printf("Itr %d : newton method is convergence !\n", itr);

    delete []delta_x;
}

