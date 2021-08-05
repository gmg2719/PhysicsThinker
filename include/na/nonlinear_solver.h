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
#ifndef _NONLINEAR_SOLVER_H_
#define _NONLINEAR_SOLVER_H_       1

#include <cstdio>
#include <cstdlib>
#include "my_vector.h"
#include "my_matrix.h"

#define MAX_ITERS_NLSOLVER          1000

// The callback function is : x = f(x) (used for the fixed point method) or F(x) = 0 (used for the non-linear equations system)
typedef void (*NL_Func_double)(int16_t n, double *x);
typedef void (*NL_Func_float)(int16_t n, float *x);
// The callback function is : G(x) = F(x) - x = 0, D(x) = G'(x), the 1-order derivative function
typedef void (*NL_DeriveFunc_double)(int16_t n, double *x);
typedef void (*NL_DeriveFunc_float)(int16_t n, float *x);

inline void na_generate_jacobi_matrix(NL_Func_float func, int16_t n, float *x, Matrix<float>& jacobi_matrix)
{
    if (n <= 1) {
        return;
    }

    if ((jacobi_matrix.M != n) || (jacobi_matrix.N != n)) {
        jacobi_matrix.resize(n, n);
    }

    float *x_fwd = new float[n];
    float *fx = new float[n];

    // Calculate the F(x1, x2, x3, ..., xn)
    for (int16_t k = 0; k < n; k++) {
        fx[k] = x[k];
    }
    func(n, fx);

    for (int16_t i = 0; i < n; i++) {
        for (int16_t k = 0; k < n; k++) {
            x_fwd[k] = x[k];
        }
        x_fwd[i] += 1E-4;
        func(n, x_fwd);
        for (int16_t row = 0; row < n; row++)
        {
            jacobi_matrix(row, i) = (x_fwd[row] - fx[row]) / 1E-4;
        }
    }

    delete []fx;
    delete []x_fwd;
}

inline void na_generate_jacobi_matrix(NL_Func_double func, int16_t n, double *x, Matrix<double>& jacobi_matrix)
{
    if (n <= 1) {
        return;
    }

    if ((jacobi_matrix.M != n) || (jacobi_matrix.N != n)) {
        jacobi_matrix.resize(n, n);
    }

    double *x_fwd = new double[n];
    double *fx = new double[n];

    // Calculate the F(x1, x2, x3, ..., xn)
    for (int16_t k = 0; k < n; k++) {
        fx[k] = x[k];
    }
    func(n, fx);

    for (int16_t i = 0; i < n; i++) {
        for (int16_t k = 0; k < n; k++) {
            x_fwd[k] = x[k];
        }
        x_fwd[i] += 1E-4;
        func(n, x_fwd);
        for (int16_t row = 0; row < n; row++)
        {
            jacobi_matrix(row, i) = (x_fwd[row] - fx[row]) / 1E-4;
        }
    }

    delete []fx;
    delete []x_fwd;
}

// Iterative methods
void na_nonlinear_solver_fixedpoint(NL_Func_double func, int16_t n, double *x);
void na_nonlinear_solver_fixedpoint(NL_Func_float func, int16_t n, float *x);

void na_nonlinear_solver_newton(NL_DeriveFunc_double func, int16_t n, double *x);
void na_nonlinear_solver_newton(NL_DeriveFunc_float func, int16_t n, float *x);

void na_nonlinear_solver_newtonfree(NL_Func_double func, int16_t n, double *x);
void na_nonlinear_solver_newtonfree(NL_Func_float func, int16_t n, float *x);

#endif
