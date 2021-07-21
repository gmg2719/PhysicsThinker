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
#include <utility>
#include "na/linear_solver.h"

/////////////////////////////////////
//                                 //
//          Direct methods         //
//                                 //
/////////////////////////////////////
void na_linear_solver_gauss(Matrix<float> &A, float *x, float *b)
{
    int8_t swap_flag;
    float max_var, tmp;
    int32_t max_row, dia, row_A(A.M), col_A(A.N);

    if (A.M != A.N) {
        fprintf(stderr, "Ax = b, must dim1() == dim2() !\n");
        return;
    }

    if ((x == NULL) || (b == NULL)) {
        fprintf(stderr, "Wrong input for na_linear_solver_gauss() !\n");
        return;
    }

    for (int32_t i = 0; i < row_A; i++)
    {
        max_row = i;
        max_var = A(i, i);

        swap_flag = 0;
        for (int32_t row = i + 1; row < row_A; row++)
        {
            if ((fabs(A(row, i)) > max_var) && (fabs(A(row, i)) != 0)) {
                max_row = row;
                max_var = tmp;
                swap_flag = 1;
            }
        }

        // Swap the row i <------> max_row
        if (swap_flag == 1)
        {
            std::swap(b[i], b[max_row]);
            for (int32_t k = 0; k < col_A; k++) {
                std::swap(A(i, k), A(max_row, k));
            }
        }

        if (fabs(A(i, i)) == 0) {
            fprintf(stderr, "na_linear_solver_gauss() matrix is singular !\n");
            return;
        }

        for (int32_t row = i + 1; row < col_A; row++)
        {
            tmp = A(row, i) / A(i, i);

            for (int32_t col = i + 1; col < col_A; col++) {
                A(row, col) -= tmp * A(i, col);
            }
            A(row, i) = 0;
            b[row] -= tmp * b[i];
        }
    }

    // Get the x from the backward substitution
    for (int32_t row = row_A - 1; row >= 0; row--)
    {
        tmp = b[row];
        for (int32_t col = col_A - 1; col > row; col--) {
            tmp -= x[col] * A(row, col);
        }
        x[row] = tmp / A(row, row);
    }
}

void na_linear_solver_gauss(Matrix<double> &A, double *x, double *b)
{
    int8_t swap_flag;
    double max_var, tmp;
    int32_t max_row, dia, row_A(A.M), col_A(A.N);

    if (A.M != A.N) {
        fprintf(stderr, "Ax = b, must dim1() == dim2() !\n");
        return;
    }

    if ((x == NULL) || (b == NULL)) {
        fprintf(stderr, "Wrong input for na_linear_solver_gauss() !\n");
        return;
    }

    for (int32_t i = 0; i < row_A; i++)
    {
        max_row = i;
        max_var = A(i, i);

        swap_flag = 0;
        for (int32_t row = i + 1; row < row_A; row++)
        {
            if ((fabs(A(row, i)) > max_var) && (fabs(A(row, i)) != 0)) {
                max_row = row;
                max_var = tmp;
                swap_flag = 1;
            }
        }

        // Swap the row i <------> max_row
        if (swap_flag == 1)
        {
            std::swap(b[i], b[max_row]);
            for (int32_t k = 0; k < col_A; k++) {
                std::swap(A(i, k), A(max_row, k));
            }
        }

        if (fabs(A(i, i)) == 0) {
            fprintf(stderr, "na_linear_solver_gauss() matrix is singular !\n");
            return;
        }

        for (int32_t row = i + 1; row < col_A; row++)
        {
            tmp = A(row, i) / A(i, i);

            for (int32_t col = i + 1; col < col_A; col++) {
                A(row, col) -= tmp * A(i, col);
            }
            A(row, i) = 0;
            b[row] -= tmp * b[i];
        }
    }

    // Get the x from the backward substitution
    for (int32_t row = row_A - 1; row >= 0; row--)
    {
        tmp = b[row];
        for (int32_t col = col_A - 1; col > row; col--) {
            tmp -= x[col] * A(row, col);
        }
        x[row] = tmp / A(row, row);
    }
}

/////////////////////////////////////
//                                 //
//        Iterative methods        //
//                                 //
/////////////////////////////////////

