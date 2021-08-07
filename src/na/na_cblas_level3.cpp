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
#include "na/na_cblas.h"
#include "na/na_cblas_error.h"

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int ldf, ldg;
  int TransF, TransG;
  const float *F, *G;

  CHECK_ARGS14(GEMM,Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    F = A;
    ldf = lda;
    TransF = (TransA == CblasConjTrans) ? CblasTrans : TransA;
    G = B;
    ldg = ldb;
    TransG = (TransB == CblasConjTrans) ? CblasTrans : TransB;
  } else {
    n1 = N;
    n2 = M;
    F = B;
    ldf = ldb;
    TransF = (TransB == CblasConjTrans) ? CblasTrans : TransB;
    G = A;
    ldg = lda;
    TransG = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        C[ldc * i + j] = 0.0;
      }
    }
  } else if (beta != 1.0) {
    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        C[ldc * i + j] *= beta;
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (TransF == CblasNoTrans && TransG == CblasNoTrans) {
    /* form  C := alpha*A*B + C */
    for (k = 0; k < K; k++) {
      for (i = 0; i < n1; i++) {
        const float temp = alpha * F[ldf * i + k];
        if (temp != 0.0) {
          for (j = 0; j < n2; j++) {
            C[ldc * i + j] += temp * G[ldg * k + j];
          }
        }
      }
    }
  } else if (TransF == CblasNoTrans && TransG == CblasTrans) {
    /* form  C := alpha*A*B' + C */
    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += F[ldf * i + k] * G[ldg * j + k];
        }
        C[ldc * i + j] += alpha * temp;
      }
    }
  } else if (TransF == CblasTrans && TransG == CblasNoTrans) {
    for (k = 0; k < K; k++) {
      for (i = 0; i < n1; i++) {
        const float temp = alpha * F[ldf * k + i];
        if (temp != 0.0) {
          for (j = 0; j < n2; j++) {
            C[ldc * i + j] += temp * G[ldg * k + j];
          }
        }
      }
    }

  } else if (TransF == CblasTrans && TransG == CblasTrans) {

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += F[ldf * k + i] * G[ldg * j + k];
        }
        C[ldc * i + j] += alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_sgemm()\n");
  }
}

void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta,
                 float *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int uplo, side;

  CHECK_ARGS13(SYMM,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    uplo = Uplo;
    side = Side;
  } else {
    n1 = N;
    n2 = M;
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    side = (Side == CblasLeft) ? CblasRight : CblasLeft;
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        C[ldc * i + j] = 0.0;
      }
    }
  } else if (beta != 1.0) {
    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        C[ldc * i + j] *= beta;
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (side == CblasLeft && uplo == CblasUpper) {

    /* form  C := alpha*A*B + C */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        const float temp1 = alpha * B[ldb * i + j];
        float temp2 = 0.0;
        C[i * ldc + j] += temp1 * A[i * lda + i];
        for (k = i + 1; k < n1; k++) {
          const float Aik = A[i * lda + k];
          C[k * ldc + j] += Aik * temp1;
          temp2 += Aik * B[ldb * k + j];
        }
        C[i * ldc + j] += alpha * temp2;
      }
    }

  } else if (side == CblasLeft && uplo == CblasLower) {

    /* form  C := alpha*A*B + C */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        const float temp1 = alpha * B[ldb * i + j];
        float temp2 = 0.0;
        for (k = 0; k < i; k++) {
          const float Aik = A[i * lda + k];
          C[k * ldc + j] += Aik * temp1;
          temp2 += Aik * B[ldb * k + j];
        }
        C[i * ldc + j] += temp1 * A[i * lda + i] + alpha * temp2;
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper) {

    /* form  C := alpha*B*A + C */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        const float temp1 = alpha * B[ldb * i + j];
        float temp2 = 0.0;
        C[i * ldc + j] += temp1 * A[j * lda + j];
        for (k = j + 1; k < n2; k++) {
          const float Ajk = A[j * lda + k];
          C[i * ldc + k] += temp1 * Ajk;
          temp2 += B[ldb * i + k] * Ajk;
        }
        C[i * ldc + j] += alpha * temp2;
      }
    }

  } else if (side == CblasRight && uplo == CblasLower) {

    /* form  C := alpha*B*A + C */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        const float temp1 = alpha * B[ldb * i + j];
        float temp2 = 0.0;
        for (k = 0; k < j; k++) {
          const float Ajk = A[j * lda + k];
          C[i * ldc + k] += temp1 * Ajk;
          temp2 += B[ldb * i + k] * Ajk;
        }
        C[i * ldc + j] += temp1 * A[j * lda + j] + alpha * temp2;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssymm()\n");
  }
}

void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float beta, float *C, const int ldc)
{
}

void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const float alpha, const float *A, const int lda,
                  const float *B, const int ldb, const float beta,
                  float *C, const int ldc)
{
}

void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb)
{
}

void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb)
{
}

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc)
{
}

void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta,
                 double *C, const int ldc)
{
}

void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc)
{
}

void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const double alpha, const double *A, const int lda,
                  const double *B, const int ldb, const double beta,
                  double *C, const int ldc)
{
}

void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
{
}

void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
{
}


void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
}

void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
}

void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *beta, void *C, const int ldc)
{
}

void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const void *beta,
                  void *C, const int ldc)
{
}

void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
}


void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
}

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
}


void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
}

void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *beta, void *C, const int ldc)
{
}


void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const void *beta,
                  void *C, const int ldc)
{
}

void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
}


void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
}


void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
}

void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const void *A, const int lda,
                 const float beta, void *C, const int ldc)
{
}


void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const float beta,
                  void *C, const int ldc)
{
}

void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
}

void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const void *A, const int lda,
                 const double beta, void *C, const int ldc)
{
}


void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const double beta,
                  void *C, const int ldc)
{
}

