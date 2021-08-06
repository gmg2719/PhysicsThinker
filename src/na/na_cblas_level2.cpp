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
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS10 (TBMV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

    if (N == 0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x := A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            float temp = (nonunit ? A[lda * i + 0] : 1.0) * X[ix];
            const int j_min = i + 1;
            const int j_max = NA_MIN(N, i + K + 1);
            int jx = OFFSET(N, incX) + j_min * incX;

            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + (j - i)];
                jx += incX;
            }

            X[ix] = temp;
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float temp = (nonunit ? A[lda * i + K] : 1.0) * X[ix];
            const int j_min = (i > K ? i - K : 0);
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + (K - i + j)];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := A'*x */
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float temp = 0.0;
            const int j_min = (K > i ? 0 : i - K);
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + (i - j)];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + 0];
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
            const int j_max = NA_MIN(N, i + K + 1);
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + (K - j + i)];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + K];
            } else {
                X[ix] += temp;
            }
            ix += incX;
        }
    }
}


void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS8(TPMV,order,Uplo,TransA,Diag,N,Ap,X,incX);

    if (N == 0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x:= A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            float atmp = Ap[TPUP(N, i, i)];
            float temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX) + (i + 1) * incX;
            for (int j = i + 1; j < N; j++) {
                atmp = Ap[TPUP(N, i, j)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float atmp = Ap[TPLO(N, i, i)];
            float temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                atmp = Ap[TPLO(N, i, j)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := A'*x */
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            float atmp = Ap[TPUP(N, i, i)];
            float temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                atmp = Ap[TPUP(N, j, i)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            float atmp = Ap[TPLO(N, i, i)];
            float temp = (nonunit ? X[ix] * atmp : X[ix]);
            int  jx = OFFSET(N, incX) + (i + 1) * incX;
            for (int j = i + 1; j < N; j++) {
                atmp = Ap[TPLO(N, j, i)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix += incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_stpmv()\n");
    }
}


void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, float *X,
                 const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    int ix, jx;
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS9(TRSV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

    if (N == 0)
        return;

    /* form  x := inv( A )*x */
    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* backsubstitution */
        ix = OFFSET(N, incX) + incX * (N - 1);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * (N - 1) + (N - 1)];
        }
        ix -= incX;
        for (int i = N - 1; i > 0 && i--;) {
            float tmp = X[ix];
            jx = ix + incX;
            for (int j = i + 1; j < N; j++) {
                const float Aij = A[lda * i + j];
                tmp -= Aij * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        /* forward substitution */
        ix = OFFSET(N, incX);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * 0 + 0];
        }
        ix += incX;
        for (int i = 1; i < N; i++) {
            float tmp = X[ix];
            jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                const float Aij = A[lda * i + j];
                tmp -= Aij * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := inv( A' )*x */
        /* forward substitution */
        ix = OFFSET(N, incX);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * 0 + 0];
        }
        ix += incX;
        for (int i = 1; i < N; i++) {
            float tmp = X[ix];
            jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                const float Aji = A[lda * j + i];
                tmp -= Aji * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        /* backsubstitution */
        ix = OFFSET(N, incX) + (N - 1) * incX;
        if (nonunit) {
            X[ix] = X[ix] / A[lda * (N - 1) + (N - 1)];
        }
        ix -= incX;
        for (int i = N - 1; i > 0 && i--;) {
            float tmp = X[ix];
            jx = ix + incX;
            for (int j = i + 1; j < N; j++) {
                const float Aji = A[lda * j + i];
                tmp -= Aji * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix -= incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_strsv()\n");
    }
}

void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda,
                 float *X, const int incX)
{
  const int nonunit = (Diag == CblasNonUnit);
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  CHECK_ARGS10(TBSV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (int i = N; i > 0 && i--;) {
      float tmp = X[ix];
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const float Aij = A[lda * i + (j - i)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[lda * i + 0];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    /* forward substitution */
    int ix = OFFSET(N, incX);

    for (int i = 0; i < N; i++) {
      float tmp = X[ix];
      const int j_min = (i > K ? i - K : 0);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const float Aij = A[lda * i + (K + j - i)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[lda * i + K];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {

    /* form  x := inv( A' )*x */
    /* forward substitution */
    int ix = OFFSET(N, incX);
    for (int i = 0; i < N; i++) {
      float tmp = X[ix];
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const float Aji = A[(i - j) + lda * j];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[0 + lda * i];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (int i = N; i > 0 && i--;) {
      float tmp = X[ix];
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const float Aji = A[(K + i - j) + lda * j];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[K + lda * i];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_stbsv()\n");
  }
}


void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX)
{
  const int nonunit = (Diag == CblasNonUnit);
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  CHECK_ARGS8(TPSV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + incX * (N - 1);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPUP(N, (N - 1), (N - 1))];
    }
    ix -= incX;
    for (int i = N - 1; i > 0 && i--;) {
      float tmp = X[ix];
      int jx = ix + incX;
      for (int j = i + 1; j < N; j++) {
        const float Aij = Ap[TPUP(N, i, j)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPUP(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    /* forward substitution */
    int ix = OFFSET(N, incX);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPLO(N, 0, 0)];
    }
    ix += incX;
    for (int i = 1, j; i < N; i++) {
      float tmp = X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float Aij = Ap[TPLO(N, i, j)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPLO(N, i, j)];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {

    /* form  x := inv( A' )*x */

    /* forward substitution */
    int ix = OFFSET(N, incX);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPUP(N, 0, 0)];
    }
    ix += incX;
    for (int i = 1; i < N; i++) {
      float tmp = X[ix];
      int jx = OFFSET(N, incX);
      for (int j = 0; j < i; j++) {
        const float Aji = Ap[TPUP(N, j, i)];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPUP(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPLO(N, (N - 1), (N - 1))];
    }
    ix -= incX;
    for (int i = N - 1; i > 0 && i--;) {
      float tmp = X[ix];
      int jx = ix + incX;
      for (int j = i + 1; j < N; j++) {
        const float Aji = Ap[TPLO(N, j, i)];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPLO(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_stpsv()\n");
  }
}

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
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
            double temp = 0.0;
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
            const double temp = alpha * X[ix];
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
        fprintf(stderr, "unrecognized operation for cblas_dgemv()\n");
  }
}

void cblas_dgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha,
                 const double *A, const int lda, const double *X,
                 const int incX, const double beta, double *Y, const int incY)
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
            double temp = 0.0;
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
            const double temp = alpha * X[jx];
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
        fprintf(stderr, "unrecognized operation for cblas_dgbmv()\n");
    }
}

void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, 
                 double *X, const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS9(TRMV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x := A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            double temp = 0.0;
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
            double temp = 0.0;
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
            double temp = 0.0;
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
            double temp = 0.0;
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
        fprintf(stderr, "unrecognized operation for cblas_dtrmv()\n");
    }
}

void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, 
                 double *X, const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS10 (TBMV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

    if (N == 0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x := A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            double temp = (nonunit ? A[lda * i + 0] : 1.0) * X[ix];
            const int j_min = i + 1;
            const int j_max = NA_MIN(N, i + K + 1);
            int jx = OFFSET(N, incX) + j_min * incX;

            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + (j - i)];
                jx += incX;
            }

            X[ix] = temp;
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            double temp = (nonunit ? A[lda * i + K] : 1.0) * X[ix];
            const int j_min = (i > K ? i - K : 0);
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * i + (K - i + j)];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := A'*x */
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            double temp = 0.0;
            const int j_min = (K > i ? 0 : i - K);
            const int j_max = i;
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + (i - j)];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + 0];
            } else {
                X[ix] += temp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            double temp = 0.0;
            const int j_min = i + 1;
            const int j_max = NA_MIN(N, i + K + 1);
            int jx = OFFSET(N, incX) + j_min * incX;
            for (int j = j_min; j < j_max; j++) {
                temp += X[jx] * A[lda * j + (K - j + i)];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = temp + X[ix] * A[lda * i + K];
            } else {
                X[ix] += temp;
            }
            ix += incX;
        }
    }
}

void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS8(TPMV,order,Uplo,TransA,Diag,N,Ap,X,incX);

    if (N == 0)
        return;

    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* form  x:= A*x */
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            double atmp = Ap[TPUP(N, i, i)];
            double temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX) + (i + 1) * incX;
            for (int j = i + 1; j < N; j++) {
                atmp = Ap[TPUP(N, i, j)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            double atmp = Ap[TPLO(N, i, i)];
            double temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                atmp = Ap[TPLO(N, i, j)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := A'*x */
        int ix = OFFSET(N, incX) + (N - 1) * incX;
        for (int i = N; i > 0 && i--;) {
            double atmp = Ap[TPUP(N, i, i)];
            double temp = (nonunit ? X[ix] * atmp : X[ix]);
            int jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                atmp = Ap[TPUP(N, j, i)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        int ix = OFFSET(N, incX);
        for (int i = 0; i < N; i++) {
            double atmp = Ap[TPLO(N, i, i)];
            double temp = (nonunit ? X[ix] * atmp : X[ix]);
            int  jx = OFFSET(N, incX) + (i + 1) * incX;
            for (int j = i + 1; j < N; j++) {
                atmp = Ap[TPLO(N, j, i)];
                temp += atmp * X[jx];
                jx += incX;
            }
            X[ix] = temp;
            ix += incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_dtpmv()\n");
    }
}

void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, double *X,
                 const int incX)
{
    const int nonunit = (Diag == CblasNonUnit);
    int ix, jx;
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    CHECK_ARGS9(TRSV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

    if (N == 0)
        return;

    /* form  x := inv( A )*x */
    if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper) || 
        (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
        /* backsubstitution */
        ix = OFFSET(N, incX) + incX * (N - 1);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * (N - 1) + (N - 1)];
        }
        ix -= incX;
        for (int i = N - 1; i > 0 && i--;) {
            double tmp = X[ix];
            jx = ix + incX;
            for (int j = i + 1; j < N; j++) {
                const double Aij = A[lda * i + j];
                tmp -= Aij * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix -= incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
        /* forward substitution */
        ix = OFFSET(N, incX);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * 0 + 0];
        }
        ix += incX;
        for (int i = 1; i < N; i++) {
            double tmp = X[ix];
            jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                const double Aij = A[lda * i + j];
                tmp -= Aij * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
        /* form  x := inv( A' )*x */
        /* forward substitution */
        ix = OFFSET(N, incX);
        if (nonunit) {
            X[ix] = X[ix] / A[lda * 0 + 0];
        }
        ix += incX;
        for (int i = 1; i < N; i++) {
            double tmp = X[ix];
            jx = OFFSET(N, incX);
            for (int j = 0; j < i; j++) {
                const double Aji = A[lda * j + i];
                tmp -= Aji * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix += incX;
        }
    } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower) || 
               (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
        /* backsubstitution */
        ix = OFFSET(N, incX) + (N - 1) * incX;
        if (nonunit) {
            X[ix] = X[ix] / A[lda * (N - 1) + (N - 1)];
        }
        ix -= incX;
        for (int i = N - 1; i > 0 && i--;) {
            double tmp = X[ix];
            jx = ix + incX;
            for (int j = i + 1; j < N; j++) {
                const double Aji = A[lda * j + i];
                tmp -= Aji * X[jx];
                jx += incX;
            }
            if (nonunit) {
                X[ix] = tmp / A[lda * i + i];
            } else {
                X[ix] = tmp;
            }
            ix -= incX;
        }
    } else {
        fprintf(stderr, "unrecognized operation for cblas_dtrsv()\n");
    }
}

void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda,
                 double *X, const int incX)
{
  const int nonunit = (Diag == CblasNonUnit);
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  CHECK_ARGS10(TBSV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (int i = N; i > 0 && i--;) {
      double tmp = X[ix];
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const double Aij = A[lda * i + (j - i)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[lda * i + 0];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    /* forward substitution */
    int ix = OFFSET(N, incX);

    for (int i = 0; i < N; i++) {
      double tmp = X[ix];
      const int j_min = (i > K ? i - K : 0);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const double Aij = A[lda * i + (K + j - i)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[lda * i + K];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {

    /* form  x := inv( A' )*x */
    /* forward substitution */
    int ix = OFFSET(N, incX);
    for (int i = 0; i < N; i++) {
      double tmp = X[ix];
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const double Aji = A[(i - j) + lda * j];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[0 + lda * i];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (int i = N; i > 0 && i--;) {
      double tmp = X[ix];
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (int j = j_min; j < j_max; j++) {
        const double Aji = A[(K + i - j) + lda * j];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / A[K + lda * i];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dtbsv()\n");
  }
}

void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX)
{
  const int nonunit = (Diag == CblasNonUnit);
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  CHECK_ARGS8(TPSV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* backsubstitution */
    int ix = OFFSET(N, incX) + incX * (N - 1);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPUP(N, (N - 1), (N - 1))];
    }
    ix -= incX;
    for (int i = N - 1; i > 0 && i--;) {
      double tmp = X[ix];
      int jx = ix + incX;
      for (int j = i + 1; j < N; j++) {
        const double Aij = Ap[TPUP(N, i, j)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPUP(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    /* forward substitution */
    int ix = OFFSET(N, incX);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPLO(N, 0, 0)];
    }
    ix += incX;
    for (int i = 1, j; i < N; i++) {
      double tmp = X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double Aij = Ap[TPLO(N, i, j)];
        tmp -= Aij * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPLO(N, i, j)];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {

    /* form  x := inv( A' )*x */

    /* forward substitution */
    int ix = OFFSET(N, incX);
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPUP(N, 0, 0)];
    }
    ix += incX;
    for (int i = 1; i < N; i++) {
      double tmp = X[ix];
      int jx = OFFSET(N, incX);
      for (int j = 0; j < i; j++) {
        const double Aji = Ap[TPUP(N, j, i)];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPUP(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    if (nonunit) {
      X[ix] = X[ix] / Ap[TPLO(N, (N - 1), (N - 1))];
    }
    ix -= incX;
    for (int i = N - 1; i > 0 && i--;) {
      double tmp = X[ix];
      int jx = ix + incX;
      for (int j = i + 1; j < N; j++) {
        const double Aji = Ap[TPLO(N, j, i)];
        tmp -= Aji * X[jx];
        jx += incX;
      }
      if (nonunit) {
        X[ix] = tmp / Ap[TPLO(N, i, i)];
      } else {
        X[ix] = tmp;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dtpsv()\n");
  }
}

void cblas_cgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  int lenX, lenY;
  const float alpha_real = CONST_REAL0_FLOAT(alpha);
  const float alpha_imag = CONST_IMAG0_FLOAT(alpha);
  const float beta_real = CONST_REAL0_FLOAT(beta);
  const float beta_imag = CONST_IMAG0_FLOAT(beta);
  CHECK_ARGS12(GEMV,order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);

  if (M == 0 || N == 0)
    return;

  if ((alpha_real == 0.0 && alpha_imag == 0.0)
      && (beta_real == 1.0 && beta_imag == 0.0))
    return;

  if (TransA == CblasNoTrans) {
    lenX = N;
    lenY = M;
  } else {
    lenX = M;
    lenY = N;
  }

  /* form  y := beta*y */

  if (beta_real == 0.0 && beta_imag == 0.0) {
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      REAL_FLOAT(Y, iy) = 0.0;
      IMAG_FLOAT(Y, iy) = 0.0;
      iy += incY;
    }
  } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      const float y_real = REAL_FLOAT(Y, iy);
      const float y_imag = IMAG_FLOAT(Y, iy);
      const float tmpR = y_real * beta_real - y_imag * beta_imag;
      const float tmpI = y_real * beta_imag + y_imag * beta_real;
      REAL_FLOAT(Y, iy) = tmpR;
      IMAG_FLOAT(Y, iy) = tmpI;
      iy += incY;
    }
  }

  if (alpha_real == 0.0 && alpha_imag == 0.0)
    return;

  if ((order == CblasRowMajor && TransA == CblasNoTrans)
      || (order == CblasColMajor && TransA == CblasTrans)) {
    /* form  y := alpha*A*x + y */
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      float dotR = 0.0;
      float dotI = 0.0;
      int ix = OFFSET(lenX, incX);
      for (int j = 0; j < lenX; j++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float A_imag = CONST_IMAG_FLOAT(A, lda * i + j);

        dotR += A_real * x_real - A_imag * x_imag;
        dotI += A_real * x_imag + A_imag * x_real;
        ix += incX;
      }

      REAL_FLOAT(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_FLOAT(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && TransA == CblasTrans)
             || (order == CblasColMajor && TransA == CblasNoTrans)) {
    /* form  y := alpha*A'*x + y */
    int ix = OFFSET(lenX, incX);
    for (int j = 0; j < lenX; j++) {
      float x_real = CONST_REAL_FLOAT(X, ix);
      float x_imag = CONST_IMAG_FLOAT(X, ix);
      float tmpR = alpha_real * x_real - alpha_imag * x_imag;
      float tmpI = alpha_real * x_imag + alpha_imag * x_real;

      int iy = OFFSET(lenY, incY);
      for (int i = 0; i < lenY; i++) {
        const float A_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float A_imag = CONST_IMAG_FLOAT(A, lda * j + i);
        REAL_FLOAT(Y, iy) += A_real * tmpR - A_imag * tmpI;
        IMAG_FLOAT(Y, iy) += A_real * tmpI + A_imag * tmpR;
        iy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasRowMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int ix = OFFSET(lenX, incX);
    for (int j = 0; j < lenX; j++) {
      float x_real = CONST_REAL_FLOAT(X, ix);
      float x_imag = CONST_IMAG_FLOAT(X, ix);
      float tmpR = alpha_real * x_real - alpha_imag * x_imag;
      float tmpI = alpha_real * x_imag + alpha_imag * x_real;

      int iy = OFFSET(lenY, incY);
      for (int i = 0; i < lenY; i++) {
        const float A_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float A_imag = CONST_IMAG_FLOAT(A, lda * j + i);
        REAL_FLOAT(Y, iy) += A_real * tmpR - (-A_imag) * tmpI;
        IMAG_FLOAT(Y, iy) += A_real * tmpI + (-A_imag) * tmpR;
        iy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasColMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      float dotR = 0.0;
      float dotI = 0.0;
      int ix = OFFSET(lenX, incX);
      for (int j = 0; j < lenX; j++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float A_imag = CONST_IMAG_FLOAT(A, lda * i + j);

        dotR += A_real * x_real - (-A_imag) * x_imag;
        dotI += A_real * x_imag + (-A_imag) * x_real;
        ix += incX;
      }

      REAL_FLOAT(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_FLOAT(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_cgemv()\n");
  }
}


void cblas_cgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha,
                 const void *A, const int lda, const void *X,
                 const int incX, const void *beta, void *Y, const int incY)
{
  int i, j;
  int lenX, lenY, L, U;

  const float alpha_real = CONST_REAL0_FLOAT(alpha);
  const float alpha_imag = CONST_IMAG0_FLOAT(alpha);
  const float beta_real = CONST_REAL0_FLOAT(beta);
  const float beta_imag = CONST_IMAG0_FLOAT(beta);
  CHECK_ARGS14(GBMV,order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);

  if (M == 0 || N == 0)
    return;

  if ((alpha_real == 0.0 && alpha_imag == 0.0)
      && (beta_real == 1.0 && beta_imag == 0.0))
    return;

  if (TransA == CblasNoTrans) {
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
  if (beta_real == 0.0 && beta_imag == 0.0) {
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      REAL_FLOAT(Y, iy) = 0.0;
      IMAG_FLOAT(Y, iy) = 0.0;
      iy += incY;
    }
  } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      const float y_real = REAL_FLOAT(Y, iy);
      const float y_imag = IMAG_FLOAT(Y, iy);
      const float tmpR = y_real * beta_real - y_imag * beta_imag;
      const float tmpI = y_real * beta_imag + y_imag * beta_real;
      REAL_FLOAT(Y, iy) = tmpR;
      IMAG_FLOAT(Y, iy) = tmpI;
      iy += incY;
    }
  }

  if (alpha_real == 0.0 && alpha_imag == 0.0)
    return;

  if ((order == CblasRowMajor && TransA == CblasNoTrans)
      || (order == CblasColMajor && TransA == CblasTrans)) {
    /* form  y := alpha*A*x + y */
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      float dotR = 0.0;
      float dotI = 0.0;
      const int j_min = (i > L ? i - L : 0);
      const int j_max = NA_MIN(lenX, i + U + 1);
      int ix = OFFSET(lenX, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + (L + j - i));
        const float A_imag = CONST_IMAG_FLOAT(A, lda * i + (L + j - i));

        dotR += A_real * x_real - A_imag * x_imag;
        dotI += A_real * x_imag + A_imag * x_real;
        ix += incX;
      }

      REAL_FLOAT(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_FLOAT(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && TransA == CblasTrans)
             || (order == CblasColMajor && TransA == CblasNoTrans)) {
    /* form  y := alpha*A'*x + y */
    int ix = OFFSET(lenX, incX);
    for (j = 0; j < lenX; j++) {
      const float x_real = CONST_REAL_FLOAT(X, ix);
      const float x_imag = CONST_IMAG_FLOAT(X, ix);
      float tmpR = alpha_real * x_real - alpha_imag * x_imag;
      float tmpI = alpha_real * x_imag + alpha_imag * x_real;
      if (!(tmpR == 0.0 && tmpI == 0.0)) {
        const int i_min = (j > U ? j - U : 0);
        const int i_max = NA_MIN(lenY, j + L + 1);
        int iy = OFFSET(lenY, incY) + i_min * incY;
        for (i = i_min; i < i_max; i++) {
          const float A_real = CONST_REAL_FLOAT(A, lda * j + (U + i - j));
          const float A_imag = CONST_IMAG_FLOAT(A, lda * j + (U + i - j));
          REAL_FLOAT(Y, iy) += A_real * tmpR - A_imag * tmpI;
          IMAG_FLOAT(Y, iy) += A_real * tmpI + A_imag * tmpR;
          iy += incY;
        }
      }
      ix += incX;
    }
  } else if (order == CblasRowMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int ix = OFFSET(lenX, incX);
    for (j = 0; j < lenX; j++) {
      const float x_real = CONST_REAL_FLOAT(X, ix);
      const float x_imag = CONST_IMAG_FLOAT(X, ix);
      float tmpR = alpha_real * x_real - alpha_imag * x_imag;
      float tmpI = alpha_real * x_imag + alpha_imag * x_real;
      if (!(tmpR == 0.0 && tmpI == 0.0)) {
        const int i_min = (j > U ? j - U : 0);
        const int i_max = NA_MIN(lenY, j + L + 1);
        int iy = OFFSET(lenY, incY) + i_min * incY;
        for (i = i_min; i < i_max; i++) {
          const float A_real = CONST_REAL_FLOAT(A, lda * j + (U + i - j));
          const float A_imag = CONST_IMAG_FLOAT(A, lda * j + (U + i - j));
          REAL_FLOAT(Y, iy) += A_real * tmpR - (-A_imag) * tmpI;
          IMAG_FLOAT(Y, iy) += A_real * tmpI + (-A_imag) * tmpR;
          iy += incY;
        }
      }
      ix += incX;
    }
  } else if (order == CblasColMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      float dotR = 0.0;
      float dotI = 0.0;
      const int j_min = (i > L ? i - L : 0);
      const int j_max = NA_MIN(lenX, i + U + 1);
      int ix = OFFSET(lenX, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + (L + j - i));
        const float A_imag = CONST_IMAG_FLOAT(A, lda * i + (L + j - i));

        dotR += A_real * x_real - (-A_imag) * x_imag;
        dotI += A_real * x_imag + (-A_imag) * x_real;
        ix += incX;
      }

      REAL_FLOAT(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_FLOAT(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_cgbmv()\n");
  }
}

void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, 
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);

  int i, j;

  CHECK_ARGS9(TRMV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    /* form  x := A*x */
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = i + 1;
      int jx = OFFSET(N, incX) + incX * j_min;
      for (j = j_min; j < N; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX) + (N - 1) * incX;

    for (i = N; i > 0 && i--;) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_max = i;
      int jx = OFFSET(N, incX);
      for (j = 0; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (i = N; i > 0 && i--;) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_max = i;
      int jx = OFFSET(N, incX);
      for (j = 0; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * j + i);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = i + 1;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < N; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * j + i);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctrmv()\n");
  }
}

void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, 
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS10(TBMV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* form  x := A*x */

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + incX * j_min;
      for (j = j_min; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + (j - i));
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + (j - i));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + 0);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + 0);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX) + (N - 1) * incX;

    for (i = N; i > 0 && i--;) {        /*  N-1 ... 0 */
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + (K - i + j));
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + (K - i + j));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + K);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + K);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (i = N; i > 0 && i--;) {        /*  N-1 ... 0 */
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * j + (i - j));
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * j + (i - j));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + 0);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + 0);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      float temp_r = 0.0;
      float temp_i = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        const float A_real = CONST_REAL_FLOAT(A, lda * j + (K - j + i));
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * j + (K - j + i));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        const float A_real = CONST_REAL_FLOAT(A, lda * i + K);
        const float A_imag = conj * CONST_IMAG_FLOAT(A, lda * i + K);

        REAL_FLOAT(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_FLOAT(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_FLOAT(X, ix) += temp_r;
        IMAG_FLOAT(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctbmv()\n");
  }
}

void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
  int i, j;
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);

  CHECK_ARGS8(TPMV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* form  x:= A*x */

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float Aii_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, i));
      const float Aii_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, i));
      float temp_r;
      float temp_i;
      if (nonunit) {
        float x_real = REAL_FLOAT(X, ix);
        float x_imag = IMAG_FLOAT(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_FLOAT(X, ix);
        temp_i = IMAG_FLOAT(X, ix);
      }

      {
        int jx = OFFSET(N, incX) + (i + 1) * incX;
        for (j = i + 1; j < N; j++) {
          const float Aij_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, j));
          const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, j));
          float x_real = REAL_FLOAT(X, jx);
          float x_imag = IMAG_FLOAT(X, jx);
          temp_r += Aij_real * x_real - Aij_imag * x_imag;
          temp_i += Aij_real * x_imag + Aij_imag * x_real;
          jx += incX;
        }
      }

      REAL_FLOAT(X, ix) = temp_r;
      IMAG_FLOAT(X, ix) = temp_i;
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (i = N; i > 0 && i--;) {
      const float Aii_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, i));
      const float Aii_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, i));
      float temp_r;
      float temp_i;
      if (nonunit) {
        float x_real = REAL_FLOAT(X, ix);
        float x_imag = IMAG_FLOAT(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_FLOAT(X, ix);
        temp_i = IMAG_FLOAT(X, ix);
      }

      {
        int jx = OFFSET(N, incX);
        for (j = 0; j < i; j++) {
          const float Aij_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, j));
          const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, j));
          float x_real = REAL_FLOAT(X, jx);
          float x_imag = IMAG_FLOAT(X, jx);
          temp_r += Aij_real * x_real - Aij_imag * x_imag;
          temp_i += Aij_real * x_imag + Aij_imag * x_real;
          jx += incX;
        }
      }

      REAL_FLOAT(X, ix) = temp_r;
      IMAG_FLOAT(X, ix) = temp_i;
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (i = N; i > 0 && i--;) {
      const float Aii_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, i));
      const float Aii_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, i));
      float temp_r;
      float temp_i;
      if (nonunit) {
        float x_real = REAL_FLOAT(X, ix);
        float x_imag = IMAG_FLOAT(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_FLOAT(X, ix);
        temp_i = IMAG_FLOAT(X, ix);
      }
      {
        int jx = OFFSET(N, incX);
        for (j = 0; j < i; j++) {
          float x_real = REAL_FLOAT(X, jx);
          float x_imag = IMAG_FLOAT(X, jx);
          const float Aji_real = CONST_REAL_FLOAT(Ap, TPUP(N, j, i));
          const float Aji_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, j, i));
          temp_r += Aji_real * x_real - Aji_imag * x_imag;
          temp_i += Aji_real * x_imag + Aji_imag * x_real;
          jx += incX;
        }
      }

      REAL_FLOAT(X, ix) = temp_r;
      IMAG_FLOAT(X, ix) = temp_i;
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float Aii_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, i));
      const float Aii_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, i));
      float temp_r;
      float temp_i;
      if (nonunit) {
        float x_real = REAL_FLOAT(X, ix);
        float x_imag = IMAG_FLOAT(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_FLOAT(X, ix);
        temp_i = IMAG_FLOAT(X, ix);
      }
      {
        int jx = OFFSET(N, incX) + (i + 1) * incX;
        for (j = i + 1; j < N; j++) {
          float x_real = REAL_FLOAT(X, jx);
          float x_imag = IMAG_FLOAT(X, jx);
          const float Aji_real = CONST_REAL_FLOAT(Ap, TPLO(N, j, i));
          const float Aji_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, j, i));
          temp_r += Aji_real * x_real - Aji_imag * x_imag;
          temp_i += Aji_real * x_imag + Aji_imag * x_real;
          jx += incX;
        }
      }
      REAL_FLOAT(X, ix) = temp_r;
      IMAG_FLOAT(X, ix) = temp_i;
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctpmv()\n");
  }
}

void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, void *X,
                 const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;
  int ix, jx;

  CHECK_ARGS9(TRSV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(A, lda * (N - 1) + (N - 1));
      const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * (N - 1) + (N - 1));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    ix = OFFSET(N, incX);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(A, lda * 0 + 0);
      const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * 0 + 0);
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * i + j);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    ix = OFFSET(N, incX);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(A, lda * 0 + 0);
      const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * 0 + 0);
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * j + i);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(A, lda * (N - 1) + (N - 1));
      const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * (N - 1) + (N - 1));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * j + i);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * j + i);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctrsv()\n");
  }
}

void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda,
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS10(TBSV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);

    for (i = N; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * i + (j - i));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + (j - i));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + 0);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + 0);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    int ix = OFFSET(N, incX);

    for (i = 0; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, lda * i + (K + j - i));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + (K + j - i));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, lda * i + K);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, lda * i + K);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    int ix = OFFSET(N, incX);

    for (i = 0; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, (i - j) + lda * j);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, (i - j) + lda * j);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, 0 + lda * i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, 0 + lda * i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    int ix = OFFSET(N, incX) + incX * (N - 1);

    for (i = N; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const float Aij_real = CONST_REAL_FLOAT(A, (K + i - j) + lda * j);
        const float Aij_imag = conj * CONST_IMAG_FLOAT(A, (K + i - j) + lda * j);
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(A, K + lda * i);
        const float a_imag = conj * CONST_IMAG_FLOAT(A, K + lda * i);
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctbsv()\n");
  }
}

void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS8(TPSV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(Ap, TPUP(N, (N - 1), (N - 1)));
      const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, (N - 1), (N - 1)));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      int jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const float Aij_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, j));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, j));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, i));
        const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, i));
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    int ix = OFFSET(N, incX);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(Ap, TPLO(N, 0, 0));
      const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, 0, 0));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float Aij_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, j));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, j));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, i));
        const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, i));
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    int ix = OFFSET(N, incX);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(Ap, TPUP(N, 0, 0));
      const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, 0, 0));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float Aij_real = CONST_REAL_FLOAT(Ap, TPUP(N, j, i));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, j, i));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, i));
        const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, i));
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    int ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const float a_real = CONST_REAL_FLOAT(Ap, TPLO(N, (N - 1), (N - 1)));
      const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, (N - 1), (N - 1)));
      const float x_real = REAL_FLOAT(X, ix);
      const float x_imag = IMAG_FLOAT(X, ix);
      const float s = xhypot(a_real, a_imag);
      const float b_real = a_real / s;
      const float b_imag = a_imag / s;
      REAL_FLOAT(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_FLOAT(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      float tmp_real = REAL_FLOAT(X, ix);
      float tmp_imag = IMAG_FLOAT(X, ix);
      int jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const float Aij_real = CONST_REAL_FLOAT(Ap, TPLO(N, j, i));
        const float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, j, i));
        const float x_real = REAL_FLOAT(X, jx);
        const float x_imag = IMAG_FLOAT(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const float a_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, i));
        const float a_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, i));
        const float s = xhypot(a_real, a_imag);
        const float b_real = a_real / s;
        const float b_imag = a_imag / s;
        REAL_FLOAT(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_FLOAT(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_FLOAT(X, ix) = tmp_real;
        IMAG_FLOAT(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ctpsv()\n");
  }
}


void cblas_zgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  int lenX, lenY;
  const double alpha_real = CONST_REAL0_DOUBLE(alpha);
  const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);
  const double beta_real = CONST_REAL0_DOUBLE(beta);
  const double beta_imag = CONST_IMAG0_DOUBLE(beta);
  CHECK_ARGS12(GEMV,order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);

  if (M == 0 || N == 0)
    return;

  if ((alpha_real == 0.0 && alpha_imag == 0.0)
      && (beta_real == 1.0 && beta_imag == 0.0))
    return;

  if (TransA == CblasNoTrans) {
    lenX = N;
    lenY = M;
  } else {
    lenX = M;
    lenY = N;
  }

  /* form  y := beta*y */

  if (beta_real == 0.0 && beta_imag == 0.0) {
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      REAL_DOUBLE(Y, iy) = 0.0;
      IMAG_DOUBLE(Y, iy) = 0.0;
      iy += incY;
    }
  } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      const double y_real = REAL_DOUBLE(Y, iy);
      const double y_imag = IMAG_DOUBLE(Y, iy);
      const double tmpR = y_real * beta_real - y_imag * beta_imag;
      const double tmpI = y_real * beta_imag + y_imag * beta_real;
      REAL_DOUBLE(Y, iy) = tmpR;
      IMAG_DOUBLE(Y, iy) = tmpI;
      iy += incY;
    }
  }

  if (alpha_real == 0.0 && alpha_imag == 0.0)
    return;

  if ((order == CblasRowMajor && TransA == CblasNoTrans)
      || (order == CblasColMajor && TransA == CblasTrans)) {
    /* form  y := alpha*A*x + y */
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      double dotR = 0.0;
      double dotI = 0.0;
      int ix = OFFSET(lenX, incX);
      for (int j = 0; j < lenX; j++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * i + j);

        dotR += A_real * x_real - A_imag * x_imag;
        dotI += A_real * x_imag + A_imag * x_real;
        ix += incX;
      }

      REAL_DOUBLE(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_DOUBLE(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && TransA == CblasTrans)
             || (order == CblasColMajor && TransA == CblasNoTrans)) {
    /* form  y := alpha*A'*x + y */
    int ix = OFFSET(lenX, incX);
    for (int j = 0; j < lenX; j++) {
      double x_real = CONST_REAL_DOUBLE(X, ix);
      double x_imag = CONST_IMAG_DOUBLE(X, ix);
      double tmpR = alpha_real * x_real - alpha_imag * x_imag;
      double tmpI = alpha_real * x_imag + alpha_imag * x_real;

      int iy = OFFSET(lenY, incY);
      for (int i = 0; i < lenY; i++) {
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * j + i);
        REAL_DOUBLE(Y, iy) += A_real * tmpR - A_imag * tmpI;
        IMAG_DOUBLE(Y, iy) += A_real * tmpI + A_imag * tmpR;
        iy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasRowMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int ix = OFFSET(lenX, incX);
    for (int j = 0; j < lenX; j++) {
      double x_real = CONST_REAL_DOUBLE(X, ix);
      double x_imag = CONST_IMAG_DOUBLE(X, ix);
      double tmpR = alpha_real * x_real - alpha_imag * x_imag;
      double tmpI = alpha_real * x_imag + alpha_imag * x_real;

      int iy = OFFSET(lenY, incY);
      for (int i = 0; i < lenY; i++) {
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * j + i);
        REAL_DOUBLE(Y, iy) += A_real * tmpR - (-A_imag) * tmpI;
        IMAG_DOUBLE(Y, iy) += A_real * tmpI + (-A_imag) * tmpR;
        iy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasColMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int iy = OFFSET(lenY, incY);
    for (int i = 0; i < lenY; i++) {
      double dotR = 0.0;
      double dotI = 0.0;
      int ix = OFFSET(lenX, incX);
      for (int j = 0; j < lenX; j++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * i + j);

        dotR += A_real * x_real - (-A_imag) * x_imag;
        dotI += A_real * x_imag + (-A_imag) * x_real;
        ix += incX;
      }

      REAL_DOUBLE(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_DOUBLE(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_zgemv()\n");
  }
}

void cblas_zgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha,
                 const void *A, const int lda, const void *X,
                 const int incX, const void *beta, void *Y, const int incY)
{
  int i, j;
  int lenX, lenY, L, U;

  const double alpha_real = CONST_REAL0_DOUBLE(alpha);
  const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);
  const double beta_real = CONST_REAL0_DOUBLE(beta);
  const double beta_imag = CONST_IMAG0_DOUBLE(beta);
  CHECK_ARGS14(GBMV,order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);

  if (M == 0 || N == 0)
    return;

  if ((alpha_real == 0.0 && alpha_imag == 0.0)
      && (beta_real == 1.0 && beta_imag == 0.0))
    return;

  if (TransA == CblasNoTrans) {
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
  if (beta_real == 0.0 && beta_imag == 0.0) {
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      REAL_DOUBLE(Y, iy) = 0.0;
      IMAG_DOUBLE(Y, iy) = 0.0;
      iy += incY;
    }
  } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      const double y_real = REAL_DOUBLE(Y, iy);
      const double y_imag = IMAG_DOUBLE(Y, iy);
      const double tmpR = y_real * beta_real - y_imag * beta_imag;
      const double tmpI = y_real * beta_imag + y_imag * beta_real;
      REAL_DOUBLE(Y, iy) = tmpR;
      IMAG_DOUBLE(Y, iy) = tmpI;
      iy += incY;
    }
  }

  if (alpha_real == 0.0 && alpha_imag == 0.0)
    return;

  if ((order == CblasRowMajor && TransA == CblasNoTrans)
      || (order == CblasColMajor && TransA == CblasTrans)) {
    /* form  y := alpha*A*x + y */
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      double dotR = 0.0;
      double dotI = 0.0;
      const int j_min = (i > L ? i - L : 0);
      const int j_max = NA_MIN(lenX, i + U + 1);
      int ix = OFFSET(lenX, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + (L + j - i));
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * i + (L + j - i));

        dotR += A_real * x_real - A_imag * x_imag;
        dotI += A_real * x_imag + A_imag * x_real;
        ix += incX;
      }

      REAL_DOUBLE(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_DOUBLE(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && TransA == CblasTrans)
             || (order == CblasColMajor && TransA == CblasNoTrans)) {
    /* form  y := alpha*A'*x + y */
    int ix = OFFSET(lenX, incX);
    for (j = 0; j < lenX; j++) {
      const double x_real = CONST_REAL_DOUBLE(X, ix);
      const double x_imag = CONST_IMAG_DOUBLE(X, ix);
      double tmpR = alpha_real * x_real - alpha_imag * x_imag;
      double tmpI = alpha_real * x_imag + alpha_imag * x_real;
      if (!(tmpR == 0.0 && tmpI == 0.0)) {
        const int i_min = (j > U ? j - U : 0);
        const int i_max = NA_MIN(lenY, j + L + 1);
        int iy = OFFSET(lenY, incY) + i_min * incY;
        for (i = i_min; i < i_max; i++) {
          const double A_real = CONST_REAL_DOUBLE(A, lda * j + (U + i - j));
          const double A_imag = CONST_IMAG_DOUBLE(A, lda * j + (U + i - j));
          REAL_DOUBLE(Y, iy) += A_real * tmpR - A_imag * tmpI;
          IMAG_DOUBLE(Y, iy) += A_real * tmpI + A_imag * tmpR;
          iy += incY;
        }
      }
      ix += incX;
    }
  } else if (order == CblasRowMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int ix = OFFSET(lenX, incX);
    for (j = 0; j < lenX; j++) {
      const double x_real = CONST_REAL_DOUBLE(X, ix);
      const double x_imag = CONST_IMAG_DOUBLE(X, ix);
      double tmpR = alpha_real * x_real - alpha_imag * x_imag;
      double tmpI = alpha_real * x_imag + alpha_imag * x_real;
      if (!(tmpR == 0.0 && tmpI == 0.0)) {
        const int i_min = (j > U ? j - U : 0);
        const int i_max = NA_MIN(lenY, j + L + 1);
        int iy = OFFSET(lenY, incY) + i_min * incY;
        for (i = i_min; i < i_max; i++) {
          const double A_real = CONST_REAL_DOUBLE(A, lda * j + (U + i - j));
          const double A_imag = CONST_IMAG_DOUBLE(A, lda * j + (U + i - j));
          REAL_DOUBLE(Y, iy) += A_real * tmpR - (-A_imag) * tmpI;
          IMAG_DOUBLE(Y, iy) += A_real * tmpI + (-A_imag) * tmpR;
          iy += incY;
        }
      }
      ix += incX;
    }
  } else if (order == CblasColMajor && TransA == CblasConjTrans) {
    /* form  y := alpha*A^H*x + y */
    int iy = OFFSET(lenY, incY);
    for (i = 0; i < lenY; i++) {
      double dotR = 0.0;
      double dotI = 0.0;
      const int j_min = (i > L ? i - L : 0);
      const int j_max = NA_MIN(lenX, i + U + 1);
      int ix = OFFSET(lenX, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + (L + j - i));
        const double A_imag = CONST_IMAG_DOUBLE(A, lda * i + (L + j - i));

        dotR += A_real * x_real - (-A_imag) * x_imag;
        dotI += A_real * x_imag + (-A_imag) * x_real;
        ix += incX;
      }

      REAL_DOUBLE(Y, iy) += alpha_real * dotR - alpha_imag * dotI;
      IMAG_DOUBLE(Y, iy) += alpha_real * dotI + alpha_imag * dotR;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_zgbmv()\n");
  }
}

void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, 
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);

  int i, j;

  CHECK_ARGS9(TRMV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    /* form  x := A*x */
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = i + 1;
      int jx = OFFSET(N, incX) + incX * j_min;
      for (j = j_min; j < N; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX) + (N - 1) * incX;

    for (i = N; i > 0 && i--;) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_max = i;
      int jx = OFFSET(N, incX);
      for (j = 0; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (i = N; i > 0 && i--;) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_max = i;
      int jx = OFFSET(N, incX);
      for (j = 0; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + i);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = i + 1;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < N; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + i);

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztrmv()\n");
  }
}

void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, 
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS10(TBMV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* form  x := A*x */

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + incX * j_min;
      for (j = j_min; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + (j - i));
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + (j - i));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + 0);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + 0);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX) + (N - 1) * incX;

    for (i = N; i > 0 && i--;) {        /*  N-1 ... 0 */
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + (K - i + j));
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + (K - i + j));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + K);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + K);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + (N - 1) * incX;
    for (i = N; i > 0 && i--;) {        /*  N-1 ... 0 */
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + (i - j));
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + (i - j));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + 0);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + 0);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      double temp_r = 0.0;
      double temp_i = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        const double A_real = CONST_REAL_DOUBLE(A, lda * j + (K - j + i));
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + (K - j + i));

        temp_r += A_real * x_real - A_imag * x_imag;
        temp_i += A_real * x_imag + A_imag * x_real;

        jx += incX;
      }
      if (nonunit) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        const double A_real = CONST_REAL_DOUBLE(A, lda * i + K);
        const double A_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + K);

        REAL_DOUBLE(X, ix) = temp_r + (A_real * x_real - A_imag * x_imag);
        IMAG_DOUBLE(X, ix) = temp_i + (A_real * x_imag + A_imag * x_real);
      } else {
        REAL_DOUBLE(X, ix) += temp_r;
        IMAG_DOUBLE(X, ix) += temp_i;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztbmv()\n");
  }
}

void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
  int i, j;
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);

  CHECK_ARGS8(TPMV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {
    /* form  x:= A*x */

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double Aii_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, i));
      const double Aii_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, i));
      double temp_r;
      double temp_i;
      if (nonunit) {
        double x_real = REAL_DOUBLE(X, ix);
        double x_imag = IMAG_DOUBLE(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_DOUBLE(X, ix);
        temp_i = IMAG_DOUBLE(X, ix);
      }

      {
        int jx = OFFSET(N, incX) + (i + 1) * incX;
        for (j = i + 1; j < N; j++) {
          const double Aij_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, j));
          const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, j));
          double x_real = REAL_DOUBLE(X, jx);
          double x_imag = IMAG_DOUBLE(X, jx);
          temp_r += Aij_real * x_real - Aij_imag * x_imag;
          temp_i += Aij_real * x_imag + Aij_imag * x_real;
          jx += incX;
        }
      }

      REAL_DOUBLE(X, ix) = temp_r;
      IMAG_DOUBLE(X, ix) = temp_i;
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (i = N; i > 0 && i--;) {
      const double Aii_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, i));
      const double Aii_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, i));
      double temp_r;
      double temp_i;
      if (nonunit) {
        double x_real = REAL_DOUBLE(X, ix);
        double x_imag = IMAG_DOUBLE(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_DOUBLE(X, ix);
        temp_i = IMAG_DOUBLE(X, ix);
      }

      {
        int jx = OFFSET(N, incX);
        for (j = 0; j < i; j++) {
          const double Aij_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, j));
          const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, j));
          double x_real = REAL_DOUBLE(X, jx);
          double x_imag = IMAG_DOUBLE(X, jx);
          temp_r += Aij_real * x_real - Aij_imag * x_imag;
          temp_i += Aij_real * x_imag + Aij_imag * x_real;
          jx += incX;
        }
      }

      REAL_DOUBLE(X, ix) = temp_r;
      IMAG_DOUBLE(X, ix) = temp_i;
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := A'*x */

    int ix = OFFSET(N, incX) + incX * (N - 1);
    for (i = N; i > 0 && i--;) {
      const double Aii_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, i));
      const double Aii_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, i));
      double temp_r;
      double temp_i;
      if (nonunit) {
        double x_real = REAL_DOUBLE(X, ix);
        double x_imag = IMAG_DOUBLE(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_DOUBLE(X, ix);
        temp_i = IMAG_DOUBLE(X, ix);
      }
      {
        int jx = OFFSET(N, incX);
        for (j = 0; j < i; j++) {
          double x_real = REAL_DOUBLE(X, jx);
          double x_imag = IMAG_DOUBLE(X, jx);
          const double Aji_real = CONST_REAL_DOUBLE(Ap, TPUP(N, j, i));
          const double Aji_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, j, i));
          temp_r += Aji_real * x_real - Aji_imag * x_imag;
          temp_i += Aji_real * x_imag + Aji_imag * x_real;
          jx += incX;
        }
      }

      REAL_DOUBLE(X, ix) = temp_r;
      IMAG_DOUBLE(X, ix) = temp_i;
      ix -= incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double Aii_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, i));
      const double Aii_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, i));
      double temp_r;
      double temp_i;
      if (nonunit) {
        double x_real = REAL_DOUBLE(X, ix);
        double x_imag = IMAG_DOUBLE(X, ix);
        temp_r = Aii_real * x_real - Aii_imag * x_imag;
        temp_i = Aii_real * x_imag + Aii_imag * x_real;
      } else {
        temp_r = REAL_DOUBLE(X, ix);
        temp_i = IMAG_DOUBLE(X, ix);
      }
      {
        int jx = OFFSET(N, incX) + (i + 1) * incX;
        for (j = i + 1; j < N; j++) {
          double x_real = REAL_DOUBLE(X, jx);
          double x_imag = IMAG_DOUBLE(X, jx);
          const double Aji_real = CONST_REAL_DOUBLE(Ap, TPLO(N, j, i));
          const double Aji_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, j, i));
          temp_r += Aji_real * x_real - Aji_imag * x_imag;
          temp_i += Aji_real * x_imag + Aji_imag * x_real;
          jx += incX;
        }
      }
      REAL_DOUBLE(X, ix) = temp_r;
      IMAG_DOUBLE(X, ix) = temp_i;
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztpmv()\n");
  }
}


void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *A, const int lda, void *X,
                 const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;
  int ix, jx;

  CHECK_ARGS9(TRSV,order,Uplo,TransA,Diag,N,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(A, lda * (N - 1) + (N - 1));
      const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * (N - 1) + (N - 1));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    ix = OFFSET(N, incX);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(A, lda * 0 + 0);
      const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * 0 + 0);
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * i + j);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    ix = OFFSET(N, incX);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(A, lda * 0 + 0);
      const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * 0 + 0);
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + i);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(A, lda * (N - 1) + (N - 1));
      const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * (N - 1) + (N - 1));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * j + i);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + i);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztrsv()\n");
  }
}

void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda,
                 void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS10(TBSV,order,Uplo,TransA,Diag,N,K,A,lda,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);

    for (i = N; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * i + (j - i));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + (j - i));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + 0);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + 0);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    int ix = OFFSET(N, incX);

    for (i = 0; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, lda * i + (K + j - i));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + (K + j - i));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, lda * i + K);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + K);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    int ix = OFFSET(N, incX);

    for (i = 0; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      const int j_min = (K > i ? 0 : i - K);
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, (i - j) + lda * j);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, (i - j) + lda * j);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, 0 + lda * i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, 0 + lda * i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    int ix = OFFSET(N, incX) + incX * (N - 1);

    for (i = N; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      for (j = j_min; j < j_max; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(A, (K + i - j) + lda * j);
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(A, (K + i - j) + lda * j);
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(A, K + lda * i);
        const double a_imag = conj * CONST_IMAG_DOUBLE(A, K + lda * i);
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztbsv()\n");
  }
}

void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX)
{
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
  const int nonunit = (Diag == CblasNonUnit);
  int i, j;

  CHECK_ARGS8(TPSV,order,Uplo,TransA,Diag,N,Ap,X,incX);

  if (N == 0)
    return;

  /* form  x := inv( A )*x */

  if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasUpper)
      || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasLower)) {

    int ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(Ap, TPUP(N, (N - 1), (N - 1)));
      const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, (N - 1), (N - 1)));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      int jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, j));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, j));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, i));
        const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, i));
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }

  } else if ((order == CblasRowMajor && Trans == CblasNoTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasTrans && Uplo == CblasUpper)) {
    /* forward substitution */

    int ix = OFFSET(N, incX);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(Ap, TPLO(N, 0, 0));
      const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, 0, 0));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, j));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, j));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, i));
        const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, i));
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasUpper)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasLower)) {
    /* form  x := inv( A' )*x */

    /* forward substitution */

    int ix = OFFSET(N, incX);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(Ap, TPUP(N, 0, 0));
      const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, 0, 0));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix += incX;

    for (i = 1; i < N; i++) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(Ap, TPUP(N, j, i));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, j, i));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }
      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, i));
        const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, i));
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Trans == CblasTrans && Uplo == CblasLower)
             || (order == CblasColMajor && Trans == CblasNoTrans && Uplo == CblasUpper)) {

    /* backsubstitution */

    int ix = OFFSET(N, incX) + incX * (N - 1);

    if (nonunit) {
      const double a_real = CONST_REAL_DOUBLE(Ap, TPLO(N, (N - 1), (N - 1)));
      const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, (N - 1), (N - 1)));
      const double x_real = REAL_DOUBLE(X, ix);
      const double x_imag = IMAG_DOUBLE(X, ix);
      const double s = xhypot(a_real, a_imag);
      const double b_real = a_real / s;
      const double b_imag = a_imag / s;
      REAL_DOUBLE(X, ix) = (x_real * b_real + x_imag * b_imag) / s;
      IMAG_DOUBLE(X, ix) = (x_imag * b_real - b_imag * x_real) / s;
    }

    ix -= incX;

    for (i = N - 1; i > 0 && i--;) {
      double tmp_real = REAL_DOUBLE(X, ix);
      double tmp_imag = IMAG_DOUBLE(X, ix);
      int jx = ix + incX;
      for (j = i + 1; j < N; j++) {
        const double Aij_real = CONST_REAL_DOUBLE(Ap, TPLO(N, j, i));
        const double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, j, i));
        const double x_real = REAL_DOUBLE(X, jx);
        const double x_imag = IMAG_DOUBLE(X, jx);
        tmp_real -= Aij_real * x_real - Aij_imag * x_imag;
        tmp_imag -= Aij_real * x_imag + Aij_imag * x_real;
        jx += incX;
      }

      if (nonunit) {
        const double a_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, i));
        const double a_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, i));
        const double s = xhypot(a_real, a_imag);
        const double b_real = a_real / s;
        const double b_imag = a_imag / s;
        REAL_DOUBLE(X, ix) = (tmp_real * b_real + tmp_imag * b_imag) / s;
        IMAG_DOUBLE(X, ix) = (tmp_imag * b_real - tmp_real * b_imag) / s;
      } else {
        REAL_DOUBLE(X, ix) = tmp_real;
        IMAG_DOUBLE(X, ix) = tmp_imag;
      }
      ix -= incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ztpsv()\n");
  }
}

void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
  int i, j;

  CHECK_ARGS11(SD_SYMV,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      float temp1 = alpha * X[ix];
      float temp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = N;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += temp1 * A[lda * i + i];
      for (j = j_min; j < j_max; j++) {
        Y[jy] += temp1 * A[lda * i + j];
        temp2 += X[jx] * A[lda * i + j];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * temp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    int iy = OFFSET(N, incY) + (N - 1) * incY;
    for (i = N; i > 0 && i--;) {
      float temp1 = alpha * X[ix];
      float temp2 = 0.0;
      const int j_min = 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += temp1 * A[lda * i + i];
      for (j = j_min; j < j_max; j++) {
        Y[jy] += temp1 * A[lda * i + j];
        temp2 += X[jx] * A[lda * i + j];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * temp2;
      ix -= incX;
      iy -= incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssymv()\n");
  }
}

void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
  int i, j;

  CHECK_ARGS12(SD_SBMV,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);

  if (N == 0)
    return;

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (i = 0; i < N; i++) {
      float tmp1 = alpha * X[ix];
      float tmp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += tmp1 * A[0 + i * lda];
      for (j = j_min; j < j_max; j++) {
        float Aij = A[(j - i) + i * lda];
        Y[jy] += tmp1 * Aij;
        tmp2 += Aij * X[jx];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (i = 0; i < N; i++) {
      float tmp1 = alpha * X[ix];
      float tmp2 = 0.0;
      const int j_min = (i > K) ? i - K : 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      for (j = j_min; j < j_max; j++) {
        float Aij = A[(K - i + j) + i * lda];
        Y[jy] += tmp1 * Aij;
        tmp2 += Aij * X[jx];
        jx += incX;
        jy += incY;
      }
      Y[iy] += tmp1 * A[K + i * lda] + alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssbmv()\n");
  }
}

void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *Ap,
                 const float *X, const int incX,
                 const float beta, float *Y, const int incY)
{
  int i, j;

  CHECK_ARGS10(SD_SPMV,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      float tmp1 = alpha * X[ix];
      float tmp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = N;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;

      Y[iy] += tmp1 * Ap[TPUP(N, i, i)];

      for (j = j_min; j < j_max; j++) {
        const float apk = Ap[TPUP(N, i, j)];
        Y[jy] += tmp1 * apk;
        tmp2 += apk * X[jx];
        jy += incY;
        jx += incX;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      float tmp1 = alpha * X[ix];
      float tmp2 = 0.0;

      const int j_min = 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;

      Y[iy] += tmp1 * Ap[TPLO(N, i, i)];

      for (j = j_min; j < j_max; j++) {
        const float apk = Ap[TPLO(N, i, j)];
        Y[jy] += tmp1 * apk;
        tmp2 += apk * X[jx];
        jy += incY;
        jx += incX;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_sspmv()\n");
  }
}

void cblas_sger(const enum CBLAS_ORDER order, const int M, const int N,
                const float alpha, const float *X, const int incX,
                const float *Y, const int incY, float *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(SD_GER,order,M,N,alpha,X,incX,Y,incY,A,lda);

  if (order == CblasRowMajor) {
    int ix = OFFSET(M, incX);
    for (i = 0; i < M; i++) {
      const float tmp = alpha * X[ix];
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        A[lda * i + j] += Y[jy] * tmp;
        jy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasColMajor) {
    int jy = OFFSET(N, incY);
    for (j = 0; j < N; j++) {
      const float tmp = alpha * Y[jy];
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        A[i + lda * j] += X[ix] * tmp;
        ix += incX;
      }
      jy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_sger()\n");
  }
}

void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *A, const int lda)
{
  int i, j;

  CHECK_ARGS8(SD_SYR,order,Uplo,N,alpha,X,incX,A,lda);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp = alpha * X[ix];
      int jx = ix;
      for (j = i; j < N; j++) {
        A[lda * i + j] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp = alpha * X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j <= i; j++) {
        A[lda * i + j] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssyr()\n");
  }
}

void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *Ap)
{
  int i, j;

  CHECK_ARGS7(SD_SPR,order,Uplo,N,alpha,X,incX,Ap);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp = alpha * X[ix];
      int jx = ix;
      for (j = i; j < N; j++) {
        Ap[TPUP(N, i, j)] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp = alpha * X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j <= i; j++) {
        Ap[TPLO(N, i, j)] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_sspr()\n");
  }
}

void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A,
                const int lda)
{
  int i, j;

  CHECK_ARGS10(SD_SYR2,order,Uplo,N,alpha,X,incX,Y,incY,A,lda);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const float tmp1 = alpha * X[ix];
      const float tmp2 = alpha * Y[iy];
      int jx = ix;
      int jy = iy;
      for (j = i; j < N; j++) {
        A[lda * i + j] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const float tmp1 = alpha * X[ix];
      const float tmp2 = alpha * Y[iy];
      int jx = OFFSET(N, incX);
      int jy = OFFSET(N, incY);
      for (j = 0; j <= i; j++) {
        A[lda * i + j] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssyr2()\n");
  }
}

void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A)
{
  int i, j;

  CHECK_ARGS9(SD_SPR2,order,Uplo,N,alpha,X,incX,Y,incY,A);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const float tmp1 = alpha * X[ix];
      const float tmp2 = alpha * Y[iy];
      int jx = ix;
      int jy = iy;
      for (j = i; j < N; j++) {
        A[TPUP(N, i, j)] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const float tmp1 = alpha * X[ix];
      const float tmp2 = alpha * Y[iy];
      int jx = OFFSET(N, incX);
      int jy = OFFSET(N, incY);
      for (j = 0; j <= i; j++) {
        A[TPLO(N, i, j)] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_sspr2()\n");
  }
}

void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
  int i, j;

  CHECK_ARGS11(SD_SYMV,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      double temp1 = alpha * X[ix];
      double temp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = N;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += temp1 * A[lda * i + i];
      for (j = j_min; j < j_max; j++) {
        Y[jy] += temp1 * A[lda * i + j];
        temp2 += X[jx] * A[lda * i + j];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * temp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX) + (N - 1) * incX;
    int iy = OFFSET(N, incY) + (N - 1) * incY;
    for (i = N; i > 0 && i--;) {
      double temp1 = alpha * X[ix];
      double temp2 = 0.0;
      const int j_min = 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += temp1 * A[lda * i + i];
      for (j = j_min; j < j_max; j++) {
        Y[jy] += temp1 * A[lda * i + j];
        temp2 += X[jx] * A[lda * i + j];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * temp2;
      ix -= incX;
      iy -= incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsymv()\n");
  }
}

void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
  int i, j;

  CHECK_ARGS12(SD_SBMV,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);

  if (N == 0)
    return;

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (i = 0; i < N; i++) {
      double tmp1 = alpha * X[ix];
      double tmp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = NA_MIN(N, i + K + 1);
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      Y[iy] += tmp1 * A[0 + i * lda];
      for (j = j_min; j < j_max; j++) {
        double Aij = A[(j - i) + i * lda];
        Y[jy] += tmp1 * Aij;
        tmp2 += Aij * X[jx];
        jx += incX;
        jy += incY;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (i = 0; i < N; i++) {
      double tmp1 = alpha * X[ix];
      double tmp2 = 0.0;
      const int j_min = (i > K) ? i - K : 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;
      for (j = j_min; j < j_max; j++) {
        double Aij = A[(K - i + j) + i * lda];
        Y[jy] += tmp1 * Aij;
        tmp2 += Aij * X[jx];
        jx += incX;
        jy += incY;
      }
      Y[iy] += tmp1 * A[K + i * lda] + alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsbmv()\n");
  }
}

void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *Ap,
                 const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
  int i, j;

  CHECK_ARGS10(SD_SPMV,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);

  if (alpha == 0.0 && beta == 1.0)
    return;

  /* form  y := beta*y */
  if (beta == 0.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] = 0.0;
      iy += incY;
    }
  } else if (beta != 1.0) {
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      Y[iy] *= beta;
      iy += incY;
    }
  }

  if (alpha == 0.0)
    return;

  /* form  y := alpha*A*x + y */

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      double tmp1 = alpha * X[ix];
      double tmp2 = 0.0;
      const int j_min = i + 1;
      const int j_max = N;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;

      Y[iy] += tmp1 * Ap[TPUP(N, i, i)];

      for (j = j_min; j < j_max; j++) {
        const double apk = Ap[TPUP(N, i, j)];
        Y[jy] += tmp1 * apk;
        tmp2 += apk * X[jx];
        jy += incY;
        jx += incX;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      double tmp1 = alpha * X[ix];
      double tmp2 = 0.0;

      const int j_min = 0;
      const int j_max = i;
      int jx = OFFSET(N, incX) + j_min * incX;
      int jy = OFFSET(N, incY) + j_min * incY;

      Y[iy] += tmp1 * Ap[TPLO(N, i, i)];

      for (j = j_min; j < j_max; j++) {
        const double apk = Ap[TPLO(N, i, j)];
        Y[jy] += tmp1 * apk;
        tmp2 += apk * X[jx];
        jy += incY;
        jx += incX;
      }
      Y[iy] += alpha * tmp2;
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dspmv()\n");
  }
}

void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(SD_GER,order,M,N,alpha,X,incX,Y,incY,A,lda);

  if (order == CblasRowMajor) {
    int ix = OFFSET(M, incX);
    for (i = 0; i < M; i++) {
      const double tmp = alpha * X[ix];
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        A[lda * i + j] += Y[jy] * tmp;
        jy += incY;
      }
      ix += incX;
    }
  } else if (order == CblasColMajor) {
    int jy = OFFSET(N, incY);
    for (j = 0; j < N; j++) {
      const double tmp = alpha * Y[jy];
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        A[i + lda * j] += X[ix] * tmp;
        ix += incX;
      }
      jy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dger()\n");
  }
}

void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *A, const int lda)
{
  int i, j;

  CHECK_ARGS8(SD_SYR,order,Uplo,N,alpha,X,incX,A,lda);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp = alpha * X[ix];
      int jx = ix;
      for (j = i; j < N; j++) {
        A[lda * i + j] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp = alpha * X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j <= i; j++) {
        A[lda * i + j] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsyr()\n");
  }
}

void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *Ap)
{
  int i, j;

  CHECK_ARGS7(SD_SPR,order,Uplo,N,alpha,X,incX,Ap);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp = alpha * X[ix];
      int jx = ix;
      for (j = i; j < N; j++) {
        Ap[TPUP(N, i, j)] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp = alpha * X[ix];
      int jx = OFFSET(N, incX);
      for (j = 0; j <= i; j++) {
        Ap[TPLO(N, i, j)] += X[jx] * tmp;
        jx += incX;
      }
      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dspr()\n");
  }
}

void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A,
                const int lda)
{
  int i, j;

  CHECK_ARGS10(SD_SYR2,order,Uplo,N,alpha,X,incX,Y,incY,A,lda);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const double tmp1 = alpha * X[ix];
      const double tmp2 = alpha * Y[iy];
      int jx = ix;
      int jy = iy;
      for (j = i; j < N; j++) {
        A[lda * i + j] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const double tmp1 = alpha * X[ix];
      const double tmp2 = alpha * Y[iy];
      int jx = OFFSET(N, incX);
      int jy = OFFSET(N, incY);
      for (j = 0; j <= i; j++) {
        A[lda * i + j] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsyr2()\n");
  }
}

void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A)
{
  int i, j;

  CHECK_ARGS9(SD_SPR2,order,Uplo,N,alpha,X,incX,Y,incY,A);

  if (N == 0)
    return;

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const double tmp1 = alpha * X[ix];
      const double tmp2 = alpha * Y[iy];
      int jx = ix;
      int jy = iy;
      for (j = i; j < N; j++) {
        A[TPUP(N, i, j)] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      const double tmp1 = alpha * X[ix];
      const double tmp2 = alpha * Y[iy];
      int jx = OFFSET(N, incX);
      int jy = OFFSET(N, incY);
      for (j = 0; j <= i; j++) {
        A[TPLO(N, i, j)] += tmp1 * Y[jy] + tmp2 * X[jx];
        jx += incX;
        jy += incY;
      }
      ix += incX;
      iy += incY;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_dspr2()\n");
  }
}


void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  const int conj = (order == CblasColMajor) ? -1 : 1;
  int i, j;

  CHECK_ARGS11(CZ_HEMV,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_FLOAT(Y, iy) = 0.0;
        IMAG_FLOAT(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float y_real = REAL_FLOAT(Y, iy);
        const float y_imag = IMAG_FLOAT(Y, iy);
        const float tmpR = y_real * beta_real - y_imag * beta_imag;
        const float tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_FLOAT(Y, iy) = tmpR;
        IMAG_FLOAT(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = N;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
        /* Aii_imag is zero */
        REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
        IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(A, lda * i + j);
          float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {
      int ix = OFFSET(N, incX) + (N - 1) * incX;
      int iy = OFFSET(N, incY) + (N - 1) * incY;
      for (i = N; i > 0 && i--;) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = 0;
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
        /* Aii_imag is zero */
        REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
        IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;

        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(A, lda * i + j);
          float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + j);
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix -= incX;
        iy -= incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_chemv()\n");
    }
  }
}

void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;
  CHECK_ARGS12(CZ_HBMV,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if (N == 0)
      return;

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_FLOAT(Y, iy) = 0.0;
        IMAG_FLOAT(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float y_real = REAL_FLOAT(Y, iy);
        const float y_imag = IMAG_FLOAT(Y, iy);
        const float tmpR = y_real * beta_real - y_imag * beta_imag;
        const float tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_FLOAT(Y, iy) = tmpR;
        IMAG_FLOAT(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = NA_MIN(N, i + K + 1);
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        float Aii_real = CONST_REAL_FLOAT(A, lda * i + 0);
        /* Aii_imag is zero */
        REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
        IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(A, lda * i + (j - i));
          float Aij_imag = conj * CONST_IMAG_FLOAT(A, lda * i + (j - i));
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = (K > i ? 0 : i - K);
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;

        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(A, i * lda + (K - i + j));
          float Aij_imag = conj * CONST_IMAG_FLOAT(A, i * lda + (K - i + j));
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }

        {
          float Aii_real = CONST_REAL_FLOAT(A, lda * i + K);
          /* Aii_imag is zero */
          REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
          IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;
        }

        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_chbmv()\n");
    }
  }
}

void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;
  CHECK_ARGS10(CZ_HPMV,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_FLOAT(Y, iy) = 0.0;
        IMAG_FLOAT(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float y_real = REAL_FLOAT(Y, iy);
        const float y_imag = IMAG_FLOAT(Y, iy);
        const float tmpR = y_real * beta_real - y_imag * beta_imag;
        const float tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_FLOAT(Y, iy) = tmpR;
        IMAG_FLOAT(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = N;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        float Aii_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, i));
        /* Aii_imag is zero */
        REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
        IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(Ap, TPUP(N, i, j));
          float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPUP(N, i, j));
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        float x_real = CONST_REAL_FLOAT(X, ix);
        float x_imag = CONST_IMAG_FLOAT(X, ix);
        float temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        float temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        float temp2_real = 0.0;
        float temp2_imag = 0.0;
        const int j_min = 0;
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        float Aii_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, i));
        /* Aii_imag is zero */
        REAL_FLOAT(Y, iy) += temp1_real * Aii_real;
        IMAG_FLOAT(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          float Aij_real = CONST_REAL_FLOAT(Ap, TPLO(N, i, j));
          float Aij_imag = conj * CONST_IMAG_FLOAT(Ap, TPLO(N, i, j));
          REAL_FLOAT(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_FLOAT(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_FLOAT(X, jx);
          x_imag = CONST_IMAG_FLOAT(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_FLOAT(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_FLOAT(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_chpmv()\n");
    }
  }
}

void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(CZ_GERU,order,M,N,alpha,X,incX,Y,incY,A,lda);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (order == CblasRowMajor) {
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        const float X_real = CONST_REAL_FLOAT(X, ix);
        const float X_imag = CONST_IMAG_FLOAT(X, ix);
        const float tmp_real = alpha_real * X_real - alpha_imag * X_imag;
        const float tmp_imag = alpha_imag * X_real + alpha_real * X_imag;
        int jy = OFFSET(N, incY);
        for (j = 0; j < N; j++) {
          const float Y_real = CONST_REAL_FLOAT(Y, jy);
          const float Y_imag = CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(A, lda * i + j) += Y_real * tmp_real - Y_imag * tmp_imag;
          IMAG_FLOAT(A, lda * i + j) += Y_imag * tmp_real + Y_real * tmp_imag;
          jy += incY;
        }
        ix += incX;
      }
    } else if (order == CblasColMajor) {
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        const float Y_real = CONST_REAL_FLOAT(Y, jy);
        const float Y_imag = CONST_IMAG_FLOAT(Y, jy);
        const float tmp_real = alpha_real * Y_real - alpha_imag * Y_imag;
        const float tmp_imag = alpha_imag * Y_real + alpha_real * Y_imag;
        int ix = OFFSET(M, incX);
        for (i = 0; i < M; i++) {
          const float X_real = CONST_REAL_FLOAT(X, ix);
          const float X_imag = CONST_IMAG_FLOAT(X, ix);
          REAL_FLOAT(A, i + lda * j) += X_real * tmp_real - X_imag * tmp_imag;
          IMAG_FLOAT(A, i + lda * j) += X_imag * tmp_real + X_real * tmp_imag;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_cgeru()\n");
    }
  }
}

void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(CZ_GERC,order,M,N,alpha,X,incX,Y,incY,A,lda);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (order == CblasRowMajor) {
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        const float X_real = CONST_REAL_FLOAT(X, ix);
        const float X_imag = CONST_IMAG_FLOAT(X, ix);
        const float tmp_real = alpha_real * X_real - alpha_imag * X_imag;
        const float tmp_imag = alpha_imag * X_real + alpha_real * X_imag;
        int jy = OFFSET(N, incY);
        for (j = 0; j < N; j++) {
          const float Y_real = CONST_REAL_FLOAT(Y, jy);
          const float Y_imag = -CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(A, lda * i + j) += Y_real * tmp_real - Y_imag * tmp_imag;
          IMAG_FLOAT(A, lda * i + j) += Y_imag * tmp_real + Y_real * tmp_imag;
          jy += incY;
        }
        ix += incX;
      }
    } else if (order == CblasColMajor) {
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        const float Y_real = CONST_REAL_FLOAT(Y, jy);
        const float Y_imag = -CONST_IMAG_FLOAT(Y, jy);
        const float tmp_real = alpha_real * Y_real - alpha_imag * Y_imag;
        const float tmp_imag = alpha_imag * Y_real + alpha_real * Y_imag;
        int ix = OFFSET(M, incX);
        for (i = 0; i < M; i++) {
          const float X_real = CONST_REAL_FLOAT(X, ix);
          const float X_imag = CONST_IMAG_FLOAT(X, ix);
          REAL_FLOAT(A, i + lda * j) += X_real * tmp_real - X_imag * tmp_imag;
          IMAG_FLOAT(A, i + lda * j) += X_imag * tmp_real + X_real * tmp_imag;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_cgerc()\n");
    }
  }
}

void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X, const int incX,
                void *A, const int lda)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS8(CZ_HER,order,Uplo,N,alpha,X,incX,A,lda);

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp_real = alpha * CONST_REAL_FLOAT(X, ix);
      const float tmp_imag = alpha * conj * CONST_IMAG_FLOAT(X, ix);
      int jx = ix;

      {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, lda * i + i) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, lda * i + i) = 0;
        jx += incX;
      }

      for (j = i + 1; j < N; j++) {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, lda * i + j) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, lda * i + j) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp_real = alpha * CONST_REAL_FLOAT(X, ix);
      const float tmp_imag = alpha * conj * CONST_IMAG_FLOAT(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, lda * i + j) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, lda * i + j) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }

      {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, lda * i + i) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, lda * i + i) = 0;
        jx += incX;
      }

      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_cher()\n");
  }
}

void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X,
                const int incX, void *A)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS7(CZ_HPR,order,Uplo,N,alpha,X,incX,A);

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp_real = alpha * CONST_REAL_FLOAT(X, ix);
      const float tmp_imag = alpha * conj * CONST_IMAG_FLOAT(X, ix);
      int jx = ix;

      {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, TPUP(N, i, i)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, TPUP(N, i, i)) = 0;
        jx += incX;
      }

      for (j = i + 1; j < N; j++) {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, TPUP(N, i, j)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, TPUP(N, i, j)) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const float tmp_real = alpha * CONST_REAL_FLOAT(X, ix);
      const float tmp_imag = alpha * conj * CONST_IMAG_FLOAT(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, TPLO(N, i, j)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, TPLO(N, i, j)) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }

      {
        const float X_real = CONST_REAL_FLOAT(X, jx);
        const float X_imag = -conj * CONST_IMAG_FLOAT(X, jx);
        REAL_FLOAT(A, TPLO(N, i, i)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_FLOAT(A, TPLO(N, i, i)) = 0;
        jx += incX;
      }

      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_chpr()\n");
  }
}

void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS10(CZ_HER2,order,Uplo,N,alpha,X,incX,Y,incY,A,lda);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float Xi_real = CONST_REAL_FLOAT(X, ix);
        const float Xi_imag = CONST_IMAG_FLOAT(X, ix);
        /* tmp1 = alpha Xi */
        const float tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const float tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const float Yi_real = CONST_REAL_FLOAT(Y, iy);
        const float Yi_imag = CONST_IMAG_FLOAT(Y, iy);
        /* tmp2 = conj(alpha) Yi */
        const float tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const float tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = ix + incX;
        int jy = iy + incY;

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        REAL_FLOAT(A, lda * i + i) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_FLOAT(A, lda * i + i) = 0;

        for (j = i + 1; j < N; j++) {
          const float Xj_real = CONST_REAL_FLOAT(X, jx);
          const float Xj_imag = CONST_IMAG_FLOAT(X, jx);
          const float Yj_real = CONST_REAL_FLOAT(Y, jy);
          const float Yj_imag = CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(A, lda * i + j) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                   + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_FLOAT(A, lda * i + j) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float Xi_real = CONST_REAL_FLOAT(X, ix);
        const float Xi_imag = CONST_IMAG_FLOAT(X, ix);
        const float tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const float tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const float Yi_real = CONST_REAL_FLOAT(Y, iy);
        const float Yi_imag = CONST_IMAG_FLOAT(Y, iy);
        const float tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const float tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = OFFSET(N, incX);
        int jy = OFFSET(N, incY);

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        for (j = 0; j < i; j++) {
          const float Xj_real = CONST_REAL_FLOAT(X, jx);
          const float Xj_imag = CONST_IMAG_FLOAT(X, jx);
          const float Yj_real = CONST_REAL_FLOAT(Y, jy);
          const float Yj_imag = CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(A, lda * i + j) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                   + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_FLOAT(A, lda * i + j) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }

        REAL_FLOAT(A, lda * i + i) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_FLOAT(A, lda * i + i) = 0;

        ix += incX;
        iy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_cher2()\n");
    }
  }
}

void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *Ap)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS9(CZ_HPR2,order,Uplo,N,alpha,X,incX,Y,incY,Ap);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float Xi_real = CONST_REAL_FLOAT(X, ix);
        const float Xi_imag = CONST_IMAG_FLOAT(X, ix);
        /* tmp1 = alpha Xi */
        const float tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const float tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const float Yi_real = CONST_REAL_FLOAT(Y, iy);
        const float Yi_imag = CONST_IMAG_FLOAT(Y, iy);
        /* tmp2 = conj(alpha) Yi */
        const float tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const float tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = ix + incX;
        int jy = iy + incY;

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        REAL_FLOAT(Ap, TPUP(N, i, i)) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_FLOAT(Ap, TPUP(N, i, i)) = 0;

        for (j = i + 1; j < N; j++) {
          const float Xj_real = CONST_REAL_FLOAT(X, jx);
          const float Xj_imag = CONST_IMAG_FLOAT(X, jx);
          const float Yj_real = CONST_REAL_FLOAT(Y, jy);
          const float Yj_imag = CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(Ap, TPUP(N, i, j)) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                      + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_FLOAT(Ap, TPUP(N, i, j)) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const float Xi_real = CONST_REAL_FLOAT(X, ix);
        const float Xi_imag = CONST_IMAG_FLOAT(X, ix);
        const float tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const float tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const float Yi_real = CONST_REAL_FLOAT(Y, iy);
        const float Yi_imag = CONST_IMAG_FLOAT(Y, iy);
        const float tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const float tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = OFFSET(N, incX);
        int jy = OFFSET(N, incY);

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        for (j = 0; j < i; j++) {
          const float Xj_real = CONST_REAL_FLOAT(X, jx);
          const float Xj_imag = CONST_IMAG_FLOAT(X, jx);
          const float Yj_real = CONST_REAL_FLOAT(Y, jy);
          const float Yj_imag = CONST_IMAG_FLOAT(Y, jy);
          REAL_FLOAT(Ap, TPLO(N, i, j)) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                      + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_FLOAT(Ap, TPLO(N, i, j)) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }

        REAL_FLOAT(Ap, TPLO(N, i, i)) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_FLOAT(Ap, TPLO(N, i, i)) = 0;

        ix += incX;
        iy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_chpr2()\n");
    }
  }
}


void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  const int conj = (order == CblasColMajor) ? -1 : 1;
  int i, j;

  CHECK_ARGS11(CZ_HEMV,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_DOUBLE(Y, iy) = 0.0;
        IMAG_DOUBLE(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double y_real = REAL_DOUBLE(Y, iy);
        const double y_imag = IMAG_DOUBLE(Y, iy);
        const double tmpR = y_real * beta_real - y_imag * beta_imag;
        const double tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_DOUBLE(Y, iy) = tmpR;
        IMAG_DOUBLE(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = N;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
        /* Aii_imag is zero */
        REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
        IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(A, lda * i + j);
          double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {
      int ix = OFFSET(N, incX) + (N - 1) * incX;
      int iy = OFFSET(N, incY) + (N - 1) * incY;
      for (i = N; i > 0 && i--;) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = 0;
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
        /* Aii_imag is zero */
        REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
        IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;

        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(A, lda * i + j);
          double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + j);
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix -= incX;
        iy -= incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zhemv()\n");
    }
  }
}

void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;
  CHECK_ARGS12(CZ_HBMV,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if (N == 0)
      return;

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_DOUBLE(Y, iy) = 0.0;
        IMAG_DOUBLE(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double y_real = REAL_DOUBLE(Y, iy);
        const double y_imag = IMAG_DOUBLE(Y, iy);
        const double tmpR = y_real * beta_real - y_imag * beta_imag;
        const double tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_DOUBLE(Y, iy) = tmpR;
        IMAG_DOUBLE(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = NA_MIN(N, i + K + 1);
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        double Aii_real = CONST_REAL_DOUBLE(A, lda * i + 0);
        /* Aii_imag is zero */
        REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
        IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(A, lda * i + (j - i));
          double Aij_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + (j - i));
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = (K > i ? 0 : i - K);
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;

        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(A, i * lda + (K - i + j));
          double Aij_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + (K - i + j));
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }

        {
          double Aii_real = CONST_REAL_DOUBLE(A, lda * i + K);
          /* Aii_imag is zero */
          REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
          IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;
        }

        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zhbmv()\n");
    }
  }
}

void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;
  CHECK_ARGS10(CZ_HPMV,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        REAL_DOUBLE(Y, iy) = 0.0;
        IMAG_DOUBLE(Y, iy) = 0.0;
        iy += incY;
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double y_real = REAL_DOUBLE(Y, iy);
        const double y_imag = IMAG_DOUBLE(Y, iy);
        const double tmpR = y_real * beta_real - y_imag * beta_imag;
        const double tmpI = y_real * beta_imag + y_imag * beta_real;
        REAL_DOUBLE(Y, iy) = tmpR;
        IMAG_DOUBLE(Y, iy) = tmpI;
        iy += incY;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    /* form  y := alpha*A*x + y */

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = i + 1;
        const int j_max = N;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        double Aii_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, i));
        /* Aii_imag is zero */
        REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
        IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(Ap, TPUP(N, i, j));
          double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPUP(N, i, j));
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        double x_real = CONST_REAL_DOUBLE(X, ix);
        double x_imag = CONST_IMAG_DOUBLE(X, ix);
        double temp1_real = alpha_real * x_real - alpha_imag * x_imag;
        double temp1_imag = alpha_real * x_imag + alpha_imag * x_real;
        double temp2_real = 0.0;
        double temp2_imag = 0.0;
        const int j_min = 0;
        const int j_max = i;
        int jx = OFFSET(N, incX) + j_min * incX;
        int jy = OFFSET(N, incY) + j_min * incY;
        double Aii_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, i));
        /* Aii_imag is zero */
        REAL_DOUBLE(Y, iy) += temp1_real * Aii_real;
        IMAG_DOUBLE(Y, iy) += temp1_imag * Aii_real;
        for (j = j_min; j < j_max; j++) {
          double Aij_real = CONST_REAL_DOUBLE(Ap, TPLO(N, i, j));
          double Aij_imag = conj * CONST_IMAG_DOUBLE(Ap, TPLO(N, i, j));
          REAL_DOUBLE(Y, jy) += temp1_real * Aij_real - temp1_imag * (-Aij_imag);
          IMAG_DOUBLE(Y, jy) += temp1_real * (-Aij_imag) + temp1_imag * Aij_real;
          x_real = CONST_REAL_DOUBLE(X, jx);
          x_imag = CONST_IMAG_DOUBLE(X, jx);
          temp2_real += x_real * Aij_real - x_imag * Aij_imag;
          temp2_imag += x_real * Aij_imag + x_imag * Aij_real;
          jx += incX;
          jy += incY;
        }
        REAL_DOUBLE(Y, iy) += alpha_real * temp2_real - alpha_imag * temp2_imag;
        IMAG_DOUBLE(Y, iy) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        ix += incX;
        iy += incY;
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zhpmv()\n");
    }
  }
}

void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(CZ_GERU,order,M,N,alpha,X,incX,Y,incY,A,lda);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (order == CblasRowMajor) {
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        const double X_real = CONST_REAL_DOUBLE(X, ix);
        const double X_imag = CONST_IMAG_DOUBLE(X, ix);
        const double tmp_real = alpha_real * X_real - alpha_imag * X_imag;
        const double tmp_imag = alpha_imag * X_real + alpha_real * X_imag;
        int jy = OFFSET(N, incY);
        for (j = 0; j < N; j++) {
          const double Y_real = CONST_REAL_DOUBLE(Y, jy);
          const double Y_imag = CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(A, lda * i + j) += Y_real * tmp_real - Y_imag * tmp_imag;
          IMAG_DOUBLE(A, lda * i + j) += Y_imag * tmp_real + Y_real * tmp_imag;
          jy += incY;
        }
        ix += incX;
      }
    } else if (order == CblasColMajor) {
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        const double Y_real = CONST_REAL_DOUBLE(Y, jy);
        const double Y_imag = CONST_IMAG_DOUBLE(Y, jy);
        const double tmp_real = alpha_real * Y_real - alpha_imag * Y_imag;
        const double tmp_imag = alpha_imag * Y_real + alpha_real * Y_imag;
        int ix = OFFSET(M, incX);
        for (i = 0; i < M; i++) {
          const double X_real = CONST_REAL_DOUBLE(X, ix);
          const double X_imag = CONST_IMAG_DOUBLE(X, ix);
          REAL_DOUBLE(A, i + lda * j) += X_real * tmp_real - X_imag * tmp_imag;
          IMAG_DOUBLE(A, i + lda * j) += X_imag * tmp_real + X_real * tmp_imag;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zgeru()\n");
    }
  }
}

void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
  int i, j;

  CHECK_ARGS10(CZ_GERC,order,M,N,alpha,X,incX,Y,incY,A,lda);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (order == CblasRowMajor) {
      int ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        const double X_real = CONST_REAL_DOUBLE(X, ix);
        const double X_imag = CONST_IMAG_DOUBLE(X, ix);
        const double tmp_real = alpha_real * X_real - alpha_imag * X_imag;
        const double tmp_imag = alpha_imag * X_real + alpha_real * X_imag;
        int jy = OFFSET(N, incY);
        for (j = 0; j < N; j++) {
          const double Y_real = CONST_REAL_DOUBLE(Y, jy);
          const double Y_imag = -CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(A, lda * i + j) += Y_real * tmp_real - Y_imag * tmp_imag;
          IMAG_DOUBLE(A, lda * i + j) += Y_imag * tmp_real + Y_real * tmp_imag;
          jy += incY;
        }
        ix += incX;
      }
    } else if (order == CblasColMajor) {
      int jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        const double Y_real = CONST_REAL_DOUBLE(Y, jy);
        const double Y_imag = -CONST_IMAG_DOUBLE(Y, jy);
        const double tmp_real = alpha_real * Y_real - alpha_imag * Y_imag;
        const double tmp_imag = alpha_imag * Y_real + alpha_real * Y_imag;
        int ix = OFFSET(M, incX);
        for (i = 0; i < M; i++) {
          const double X_real = CONST_REAL_DOUBLE(X, ix);
          const double X_imag = CONST_IMAG_DOUBLE(X, ix);
          REAL_DOUBLE(A, i + lda * j) += X_real * tmp_real - X_imag * tmp_imag;
          IMAG_DOUBLE(A, i + lda * j) += X_imag * tmp_real + X_real * tmp_imag;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zgerc()\n");
    }
  }
}


void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X, const int incX,
                void *A, const int lda)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS8(CZ_HER,order,Uplo,N,alpha,X,incX,A,lda);

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp_real = alpha * CONST_REAL_DOUBLE(X, ix);
      const double tmp_imag = alpha * conj * CONST_IMAG_DOUBLE(X, ix);
      int jx = ix;

      {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, lda * i + i) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, lda * i + i) = 0;
        jx += incX;
      }

      for (j = i + 1; j < N; j++) {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, lda * i + j) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, lda * i + j) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp_real = alpha * CONST_REAL_DOUBLE(X, ix);
      const double tmp_imag = alpha * conj * CONST_IMAG_DOUBLE(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, lda * i + j) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, lda * i + j) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }

      {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, lda * i + i) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, lda * i + i) = 0;
        jx += incX;
      }

      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_zher()\n");
  }
}

void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const void *X,
                const int incX, void *A)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS7(CZ_HPR,order,Uplo,N,alpha,X,incX,A);

  if (alpha == 0.0)
    return;

  if ((order == CblasRowMajor && Uplo == CblasUpper)
      || (order == CblasColMajor && Uplo == CblasLower)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp_real = alpha * CONST_REAL_DOUBLE(X, ix);
      const double tmp_imag = alpha * conj * CONST_IMAG_DOUBLE(X, ix);
      int jx = ix;

      {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, TPUP(N, i, i)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, TPUP(N, i, i)) = 0;
        jx += incX;
      }

      for (j = i + 1; j < N; j++) {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, TPUP(N, i, j)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, TPUP(N, i, j)) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }
      ix += incX;
    }
  } else if ((order == CblasRowMajor && Uplo == CblasLower)
             || (order == CblasColMajor && Uplo == CblasUpper)) {
    int ix = OFFSET(N, incX);
    for (i = 0; i < N; i++) {
      const double tmp_real = alpha * CONST_REAL_DOUBLE(X, ix);
      const double tmp_imag = alpha * conj * CONST_IMAG_DOUBLE(X, ix);
      int jx = OFFSET(N, incX);
      for (j = 0; j < i; j++) {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, TPLO(N, i, j)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, TPLO(N, i, j)) += X_imag * tmp_real + X_real * tmp_imag;
        jx += incX;
      }

      {
        const double X_real = CONST_REAL_DOUBLE(X, jx);
        const double X_imag = -conj * CONST_IMAG_DOUBLE(X, jx);
        REAL_DOUBLE(A, TPLO(N, i, i)) += X_real * tmp_real - X_imag * tmp_imag;
        IMAG_DOUBLE(A, TPLO(N, i, i)) = 0;
        jx += incX;
      }

      ix += incX;
    }
  } else {
    fprintf(stderr, "unrecognized operation for cblas_zhpr()\n");
  }
}

void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS10(CZ_HER2,order,Uplo,N,alpha,X,incX,Y,incY,A,lda);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double Xi_real = CONST_REAL_DOUBLE(X, ix);
        const double Xi_imag = CONST_IMAG_DOUBLE(X, ix);
        /* tmp1 = alpha Xi */
        const double tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const double tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const double Yi_real = CONST_REAL_DOUBLE(Y, iy);
        const double Yi_imag = CONST_IMAG_DOUBLE(Y, iy);
        /* tmp2 = conj(alpha) Yi */
        const double tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const double tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = ix + incX;
        int jy = iy + incY;

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        REAL_DOUBLE(A, lda * i + i) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_DOUBLE(A, lda * i + i) = 0;

        for (j = i + 1; j < N; j++) {
          const double Xj_real = CONST_REAL_DOUBLE(X, jx);
          const double Xj_imag = CONST_IMAG_DOUBLE(X, jx);
          const double Yj_real = CONST_REAL_DOUBLE(Y, jy);
          const double Yj_imag = CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(A, lda * i + j) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                   + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_DOUBLE(A, lda * i + j) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double Xi_real = CONST_REAL_DOUBLE(X, ix);
        const double Xi_imag = CONST_IMAG_DOUBLE(X, ix);
        const double tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const double tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const double Yi_real = CONST_REAL_DOUBLE(Y, iy);
        const double Yi_imag = CONST_IMAG_DOUBLE(Y, iy);
        const double tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const double tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = OFFSET(N, incX);
        int jy = OFFSET(N, incY);

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        for (j = 0; j < i; j++) {
          const double Xj_real = CONST_REAL_DOUBLE(X, jx);
          const double Xj_imag = CONST_IMAG_DOUBLE(X, jx);
          const double Yj_real = CONST_REAL_DOUBLE(Y, jy);
          const double Yj_imag = CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(A, lda * i + j) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                   + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_DOUBLE(A, lda * i + j) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }

        REAL_DOUBLE(A, lda * i + i) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_DOUBLE(A, lda * i + i) = 0;

        ix += incX;
        iy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zher2()\n");
    }
  }
}

void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *Ap)
{
  int i, j;
  const int conj = (order == CblasColMajor) ? -1 : 1;

  CHECK_ARGS9(CZ_HPR2,order,Uplo,N,alpha,X,incX,Y,incY,Ap);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if ((order == CblasRowMajor && Uplo == CblasUpper)
        || (order == CblasColMajor && Uplo == CblasLower)) {
      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double Xi_real = CONST_REAL_DOUBLE(X, ix);
        const double Xi_imag = CONST_IMAG_DOUBLE(X, ix);
        /* tmp1 = alpha Xi */
        const double tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const double tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const double Yi_real = CONST_REAL_DOUBLE(Y, iy);
        const double Yi_imag = CONST_IMAG_DOUBLE(Y, iy);
        /* tmp2 = conj(alpha) Yi */
        const double tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const double tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = ix + incX;
        int jy = iy + incY;

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        REAL_DOUBLE(Ap, TPUP(N, i, i)) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_DOUBLE(Ap, TPUP(N, i, i)) = 0;

        for (j = i + 1; j < N; j++) {
          const double Xj_real = CONST_REAL_DOUBLE(X, jx);
          const double Xj_imag = CONST_IMAG_DOUBLE(X, jx);
          const double Yj_real = CONST_REAL_DOUBLE(Y, jy);
          const double Yj_imag = CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(Ap, TPUP(N, i, j)) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                      + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_DOUBLE(Ap, TPUP(N, i, j)) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }
        ix += incX;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Uplo == CblasLower)
               || (order == CblasColMajor && Uplo == CblasUpper)) {

      int ix = OFFSET(N, incX);
      int iy = OFFSET(N, incY);
      for (i = 0; i < N; i++) {
        const double Xi_real = CONST_REAL_DOUBLE(X, ix);
        const double Xi_imag = CONST_IMAG_DOUBLE(X, ix);
        const double tmp1_real = alpha_real * Xi_real - alpha_imag * Xi_imag;
        const double tmp1_imag = alpha_imag * Xi_real + alpha_real * Xi_imag;

        const double Yi_real = CONST_REAL_DOUBLE(Y, iy);
        const double Yi_imag = CONST_IMAG_DOUBLE(Y, iy);
        const double tmp2_real = alpha_real * Yi_real + alpha_imag * Yi_imag;
        const double tmp2_imag = -alpha_imag * Yi_real + alpha_real * Yi_imag;

        int jx = OFFSET(N, incX);
        int jy = OFFSET(N, incY);

        /* Aij = alpha*Xi*conj(Yj) + conj(alpha)*Yi*conj(Xj) */

        for (j = 0; j < i; j++) {
          const double Xj_real = CONST_REAL_DOUBLE(X, jx);
          const double Xj_imag = CONST_IMAG_DOUBLE(X, jx);
          const double Yj_real = CONST_REAL_DOUBLE(Y, jy);
          const double Yj_imag = CONST_IMAG_DOUBLE(Y, jy);
          REAL_DOUBLE(Ap, TPLO(N, i, j)) += ((tmp1_real * Yj_real + tmp1_imag * Yj_imag)
                                      + (tmp2_real * Xj_real + tmp2_imag * Xj_imag));
          IMAG_DOUBLE(Ap, TPLO(N, i, j)) +=
            conj * ((tmp1_imag * Yj_real - tmp1_real * Yj_imag) +
                    (tmp2_imag * Xj_real - tmp2_real * Xj_imag));
          jx += incX;
          jy += incY;
        }

        REAL_DOUBLE(Ap, TPLO(N, i, i)) += 2 * (tmp1_real * Yi_real + tmp1_imag * Yi_imag);
        IMAG_DOUBLE(Ap, TPLO(N, i, i)) = 0;

        ix += incX;
        iy += incY;
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zhpr2()\n");
    }
  }
}

