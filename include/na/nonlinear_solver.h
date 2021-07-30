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

#include "my_vector.h"
#include "my_matrix.h"

#define MAX_ITERS_NLSOLVER          1000

// The callback function is : x = F(x)
typedef void (*NL_Func_double)(int16_t n, double *x);
typedef void (*NL_Func_float)(int16_t n, float *x);
// The callback function is : G(x) = F(x) - x = 0, D(x) = G'(x), the 1-order derivative function
typedef void (*NL_DeriveFunc_double)(int16_t n, double *x);
typedef void (*NL_DeriveFunc_float)(int16_t n, float *x);

// Iterative methods
void na_nonlinear_solver_fixedpoint(NL_Func_double func, int16_t n, double *x);
void na_nonlinear_solver_fixedpoint(NL_Func_float func, int16_t n, float *x);

void na_nonlinear_solver_newton(NL_DeriveFunc_double func, int16_t n, double *x);
void na_nonlinear_solver_newton(NL_DeriveFunc_float func, int16_t n, float *x);

#endif
