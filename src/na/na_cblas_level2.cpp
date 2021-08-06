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
#include <cmath>
#include "na/na_cblas.h"
#include "na/na_cblas_error.h"

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY)
{
    int lenX, lenY;
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;

    CHECK_ARGS12(GEMV,order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);

    if (M == 0 || N == 0)
        return;

    if (alpha == 0.0 && beta == 1.0)
        return;

    if (Trans == CblasNoTrans) {
        lenX = N;
        lenY = M;
    } else {
        lenX = M;
        lenY = N;
    }

    /* form  y := beta*y */
    if (beta == 0.0) {
    int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            Y[iy] = 0.0;
            iy += incY;
        }
    } else if (beta != 1.0) {
        int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            Y[iy] *= beta;
            iy += incY;
        }
    }

    if (alpha == 0.0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans) || 
        (order == CblasColMajor && Trans == CblasTrans)) {
        /* form  y := alpha*A*x + y */
        int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            float temp = 0.0;
            int ix = OFFSET(lenX, incX);
            for (int j = 0; j < lenX; j++)
            {
                temp += X[ix] * A[lda * i + j];
                ix += incX;
            }
            Y[iy] += alpha * temp;
            iy += incY;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans) || 
               (order == CblasColMajor && Trans == CblasNoTrans)) {
        /* form  y := alpha*A'*x + y */
        int ix = OFFSET(lenX, incX);
        for (int j = 0; j < lenX; j++) {
            const float temp = alpha * X[ix];
            if (temp != 0.0) {
                int iy = OFFSET(lenY, incY);
                for (int i = 0; i < lenY; i++) {
                    Y[iy] += temp * A[lda * j + i];
                    iy += incY;
                }
            }
            ix += incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_sgemv()\n");
  }
}

void cblas_sgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const float alpha,
                 const float *A, const int lda, const float *X,
                 const int incX, const float beta, float *Y, const int incY)
{
    int lenX, lenY, L, U;
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS14(GBMV,order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);

    if (M == 0 || N == 0)
        return;

    if (alpha == 0.0 && beta == 1.0)
        return;

    if (Trans == CblasNoTrans) {
        lenX = N;
        lenY = M;
        L = KL;
        U = KU;
    } else {
        lenX = M;
        lenY = N;
        L = KU;
        U = KL;
    }

    /* form  y := beta*y */
    if (beta == 0.0) {
        int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            Y[iy] = 0;
            iy += incY;
        }
    } else if (beta != 1.0) {
        int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            Y[iy] *= beta;
            iy += incY;
        }
    }

    if (alpha == 0.0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans) || 
        (order == CblasColMajor && Trans == CblasTrans)) {
        /* form  y := alpha*A*x + y */
        int iy = OFFSET(lenY, incY);
        for (int i = 0; i < lenY; i++) {
            float temp = 0.0;
            const int j_min = (i > L ? i - L : 0);
            const int j_max = NA_MIN(lenX, i + U + 1);
            int jx = OFFSET(lenX, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[(L - i + j) + i * lda];
                jx += incX;
            }
            Y[iy] += alpha * temp;
            iy += incY;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans) || 
               (order == CblasColMajor && Trans == CblasNoTrans)) {
        /* form  y := alpha*A'*x + y */
        int jx = OFFSET(lenX, incX);
        for (int j = 0; j < lenX; j++) {
            const float temp = alpha * X[jx];
            if (temp != 0.0) {
                const int i_min = (j > U ? j - U : 0);
                const int i_max = NA_MIN(lenY, j + L + 1);
                int iy = OFFSET(lenY, incY) + i_min * incY;
                for (int i = i_min; i < i_max; i++) {
                    Y[iy] += temp * A[lda * j + (U + i - j)];
                    iy += incY;
                }
            }
            jx += incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_sgbmv()\n");
    }
}

void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, 
                 float *X, const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS9(TRMV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x := A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            float temp = 0.0;
            const int j_min = i + 1;
            const int j_max = N;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + j];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + i];
            } else {
                X[ix] += temp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float temp = 0.0;
            const int j_min = 0;
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + j];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + i];
            } else {
                X[ix] += temp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := A'*x */
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float temp = 0.0;
            const int j_min = 0;
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + i];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + i];
            } else {
                X[ix] += temp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            float temp = 0.0;
            const int j_min = i + 1;
            const int j_max = N;
            int jx = OFFSET(N, incX) + (i + 1) * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + i];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + i];
            } else {
                X[ix] += temp;
            }
            ix += incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_strmv()\n");
    }
}

void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, 
                 float *X, const int incX)
{
}


void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX)
{
}


void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, float *X,
                 const int incX)
{
}

void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda,
                 float *X, const int incX)
{
}


void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX)
{
}

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
{
}

void cblas_dgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha,
                 const double *A, const int lda, const double *X,
                 const int incX, const double beta, double *Y, const int incY)
{
}

void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, 
                 double *X, const int incX)
{
}

void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, 
                 double *X, const int incX)
{
}

void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX)
{
}

void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, double *X,
                 const int incX)
{
}

void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda,
                 double *X, const int incX)
{
}

void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX)
{
}

void cblas_cgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
}


void cblas_cgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha,
                 const void *A, const int lda, const void *X,
                 const int incX, const void *beta, void *Y, const int incY)
{
}

void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, 
                 void *X, const int incX)
{
}

void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, 
                 void *X, const int incX)
{
}

void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
}

void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, void *X,
                 const int incX)
{
}

void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda,
                 void *X, const int incX)
{
}

void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
}


void cblas_zgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
}

void cblas_zgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha,
                 const void *A, const int lda, const void *X,
                 const int incX, const void *beta, void *Y, const int incY)
{
}

void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, 
                 void *X, const int incX)
{
}

void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, 
                 void *X, const int incX)
{
}

void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
}


void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, void *X,
                 const int incX)
{
}

void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda,
                 void *X, const int incX)
{
}

void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
}

void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
}

void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
}

void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *Ap,
                 const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
}

void cblas_sger(const enum CBLAS_ORDER order, const int M, const int N,
                const float alpha, const float *X, const int incX,
                const float *Y, const int incY, float *A, const int lda)
{
}

void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *A, const int lda)
{
}

void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *Ap)
{
}

void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A,
                const int lda)
{
}

void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A)
{
}

void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
}

void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
}

void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *Ap,
                 const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
}

void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda)
{
}

void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *A, const int lda)
{
}

void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *Ap)
{
}

void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A,
                const int lda)
{
}

void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A)
{
}


void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
}

void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
}

void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X, const int incX,
                void *A, const int lda)
{
}

void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X,
                const int incX, void *A)
{
}

void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda)
{
}

void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *Ap)
{
}


void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
}

void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
}

void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda);


void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X, const int incX,
                void *A, const int lda)
{
}

void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X,
                const int incX, void *A)
{
}

void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda)
{
}

void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *Ap)
{
}

