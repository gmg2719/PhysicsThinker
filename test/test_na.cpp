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

#include <iostream>
#include "na/linear_solver.h"
#include "na/nonlinear_solver.h"

void linear_system_testing()
{
    float a[] = {
        1.00, 0.00, 0.00,  0.00,  0.00, 0.00,
        1.00, 0.63, 0.39,  0.25,  0.16, 0.10,
        1.00, 1.26, 1.58,  1.98,  2.49, 3.13,
        1.00, 1.88, 3.55,  6.70, 12.62, 23.80,
        1.00, 2.51, 6.32, 15.88, 39.90, 100.28,
        1.00, 3.14, 9.87, 31.01, 97.41, 306.02
    };
    float b[] = {-0.01, 0.61, 0.91, 0.99, 0.60, 0.02};
    float x[6] = {0.0};

    Matrix<float> A(6, 6, a);
    A.print();

    na_linear_solver_gauss(A, x, b);

    for (size_t i = 0; i < 6; i++) {
        printf("%.8e\n", x[i]);
    }
}

// When use non-linear solver, you should write your own x = F(x)
// F(x) is easy to write !!!
static void equation(int16_t n, double *x)
{
    if (n != 2) {
        return;
    }

    double r1_x = x[0] - 1000.0;
    double r1_y = x[1] - 1999.0;
    double r1 = sqrt(r1_x*r1_x + r1_y * r1_y);
    x[0] = -0.9889300396346599 * r1 + 2207.2860879738914;
    x[1] = -0.6110111606089454 * r1 + 1367.4624639174315;
}

// Input : x, return the dx = G'(x) = (F(x) - x)'
static void derivative_f(int16_t n, double *x)
{
    if (n != 2) {
        return;
    }

    double r1_x = x[0] - 1000.0;
    double r1_y = x[1] - 1999.0;
    double r1 = sqrt(r1_x*r1_x + r1_y * r1_y);
    double b1 = -0.9889300396346599 * r1 + 2207.2860879738914 - x[0];
    double b2 = -0.6110111606089454 * r1 + 1367.4624639174315 - x[1];
    double f11 = (-0.9889300396346599) * (1.0 / r1) * r1_x - 1.0;
    double f12 = (-0.9889300396346599) * (1.0 / r1) * r1_y;
    double f21 = (-0.6110111606089454) * (1.0 / r1) * r1_x;
    double f22 = (-0.6110111606089454) * (1.0 / r1) * r1_y - 1.0;
    my_matrix_2x2inv<double>(f11, f12, f21, f22);
    x[0] = f11 * (-b1) + f12 * (-b2);
    x[1] = f21 * (-b1) + f22 * (-b2);
}

void nonlinear_system_testing()
{
    double x[2] = {0.0, 0.0};
    na_nonlinear_solver_fixedpoint(equation, 2, x);

    printf("Results are :\n");
    printf("(%.6f, %.6f)\n", x[0], x[1]);

    double y[2] = {0.0, 0.0};
    na_nonlinear_solver_newton(derivative_f, 2, y);

    printf("Results are :\n");
    printf("(%.6f, %.6f)\n", y[0], y[1]);
}

int main(void)
{
    linear_system_testing();
    nonlinear_system_testing();

    return 0;
}

