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
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(SYRK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = (Trans == CblasConjTrans) ? CblasTrans : Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;

    if (Trans == CblasTrans || Trans == CblasConjTrans) {
      trans = CblasNoTrans;
    } else {
      trans = CblasTrans;
    }
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[i * lda + k] * A[j * lda + k];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[k * lda + i] * A[k * lda + j];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[i * lda + k] * A[j * lda + k];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[k * lda + i] * A[k * lda + j];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssyrk()\n");
  }
}

void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const float alpha, const float *A, const int lda,
                  const float *B, const int ldb, const float beta,
                  float *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(SYR2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = (Trans == CblasConjTrans) ? CblasTrans : Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;

    if (Trans == CblasTrans || Trans == CblasConjTrans) {
      trans = CblasNoTrans;
    } else {
      trans = CblasTrans;
    }
  }

  /* form  C := beta*C */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += (A[i * lda + k] * B[j * ldb + k]
                   + B[i * ldb + k] * A[j * lda + k]);
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasTrans) {

    for (k = 0; k < K; k++) {
      for (i = 0; i < N; i++) {
        float temp1 = alpha * A[k * lda + i];
        float temp2 = alpha * B[k * ldb + i];
        for (j = i; j < N; j++) {
          C[i * lda + j] += temp1 * B[k * ldb + j] + temp2 * A[k * lda + j];
        }
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {


    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        float temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += (A[i * lda + k] * B[j * ldb + k]
                   + B[i * ldb + k] * A[j * lda + k]);
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasTrans) {

    for (k = 0; k < K; k++) {
      for (i = 0; i < N; i++) {
        float temp1 = alpha * A[k * lda + i];
        float temp2 = alpha * B[k * ldb + i];
        for (j = 0; j <= i; j++) {
          C[i * lda + j] += temp1 * B[k * ldb + j] + temp2 * A[k * lda + j];
        }
      }
    }


  } else {
    fprintf(stderr, "unrecognized operation for cblas_ssyr2k()\n");
  }
}

void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  int side, uplo, trans;

  CHECK_ARGS12(TRMM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    side = Side;
    uplo = Uplo;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  } else {
    n1 = N;
    n2 = M;
    side = (Side == CblasLeft) ? CblasRight : CblasLeft;
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  }

  if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * TriU(A)*B */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        if (nonunit) {
          temp = A[i * lda + i] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = i + 1; k < n1; k++) {
          temp += A[lda * i + k] * B[k * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * (TriU(A))' *B */

    for (i = n1; i > 0 && i--;) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        for (k = 0; k < i; k++) {
          temp += A[lda * k + i] * B[k * ldb + j];
        }

        if (nonunit) {
          temp += A[i * lda + i] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * TriL(A)*B */


    for (i = n1; i > 0 && i--;) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        for (k = 0; k < i; k++) {
          temp += A[lda * i + k] * B[k * ldb + j];
        }

        if (nonunit) {
          temp += A[i * lda + i] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }



  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * TriL(A)' *B */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        if (nonunit) {
          temp = A[i * lda + i] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = i + 1; k < n1; k++) {
          temp += A[lda * k + i] * B[k * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * B * TriU(A) */

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {
        float temp = 0.0;

        for (k = 0; k < j; k++) {
          temp += A[lda * k + j] * B[i * ldb + k];
        }

        if (nonunit) {
          temp += A[j * lda + j] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * B * (TriU(A))' */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        if (nonunit) {
          temp = A[j * lda + j] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = j + 1; k < n2; k++) {
          temp += A[lda * j + k] * B[i * ldb + k];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha *B * TriL(A) */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        float temp = 0.0;

        if (nonunit) {
          temp = A[j * lda + j] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = j + 1; k < n2; k++) {
          temp += A[lda * k + j] * B[i * ldb + k];
        }


        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * B * TriL(A)' */

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {
        float temp = 0.0;

        for (k = 0; k < j; k++) {
          temp += A[lda * j + k] * B[i * ldb + k];
        }

        if (nonunit) {
          temp += A[j * lda + j] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_strmm()\n");
  }
}

void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb)
{
  int i, j, k;
  int n1, n2;
  const int nonunit = (Diag == CblasNonUnit);
  int side, uplo, trans;

  CHECK_ARGS12(TRSM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    side = Side;
    uplo = Uplo;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  } else {
    n1 = N;
    n2 = M;
    side = (Side == CblasLeft) ? CblasRight : CblasLeft;
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  }

  if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * inv(TriU(A)) *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = n1; i > 0 && i--;) {
      if (nonunit) {
        float Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = 0; k < i; k++) {
        const float Aki = A[k * lda + i];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aki * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * inv(TriU(A))' *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      if (nonunit) {
        float Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = i + 1; k < n1; k++) {
        const float Aik = A[i * lda + k];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * inv(TriL(A))*B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      if (nonunit) {
        float Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = i + 1; k < n1; k++) {
        const float Aki = A[k * lda + i];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aki * B[ldb * i + j];
        }
      }
    }


  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * TriL(A)' *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = n1; i > 0 && i--;) {
      if (nonunit) {
        float Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = 0; k < i; k++) {
        const float Aik = A[i * lda + k];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * B * inv(TriU(A)) */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        if (nonunit) {
          float Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          float Bij = B[ldb * i + j];
          for (k = j + 1; k < n2; k++) {
            B[ldb * i + k] -= A[j * lda + k] * Bij;
          }
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * B * inv(TriU(A))' */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {

        if (nonunit) {
          float Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          float Bij = B[ldb * i + j];
          for (k = 0; k < j; k++) {
            B[ldb * i + k] -= A[k * lda + j] * Bij;
          }
        }
      }
    }


  } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * B * inv(TriL(A)) */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {

        if (nonunit) {
          float Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          float Bij = B[ldb * i + j];
          for (k = 0; k < j; k++) {
            B[ldb * i + k] -= A[j * lda + k] * Bij;
          }
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * B * inv(TriL(A))' */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        if (nonunit) {
          float Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          float Bij = B[ldb * i + j];
          for (k = j + 1; k < n2; k++) {
            B[ldb * i + k] -= A[k * lda + j] * Bij;
          }
        }
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_strsm()\n");
  }
}

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int ldf, ldg;
  int TransF, TransG;
  const double *F, *G;

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
        const double temp = alpha * F[ldf * i + k];
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
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += F[ldf * i + k] * G[ldg * j + k];
        }
        C[ldc * i + j] += alpha * temp;
      }
    }
  } else if (TransF == CblasTrans && TransG == CblasNoTrans) {
    for (k = 0; k < K; k++) {
      for (i = 0; i < n1; i++) {
        const double temp = alpha * F[ldf * k + i];
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
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += F[ldf * k + i] * G[ldg * j + k];
        }
        C[ldc * i + j] += alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_dgemm()\n");
  }
}

void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta,
                 double *C, const int ldc)
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
        const double temp1 = alpha * B[ldb * i + j];
        double temp2 = 0.0;
        C[i * ldc + j] += temp1 * A[i * lda + i];
        for (k = i + 1; k < n1; k++) {
          const double Aik = A[i * lda + k];
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
        const double temp1 = alpha * B[ldb * i + j];
        double temp2 = 0.0;
        for (k = 0; k < i; k++) {
          const double Aik = A[i * lda + k];
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
        const double temp1 = alpha * B[ldb * i + j];
        double temp2 = 0.0;
        C[i * ldc + j] += temp1 * A[j * lda + j];
        for (k = j + 1; k < n2; k++) {
          const double Ajk = A[j * lda + k];
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
        const double temp1 = alpha * B[ldb * i + j];
        double temp2 = 0.0;
        for (k = 0; k < j; k++) {
          const double Ajk = A[j * lda + k];
          C[i * ldc + k] += temp1 * Ajk;
          temp2 += B[ldb * i + k] * Ajk;
        }
        C[i * ldc + j] += temp1 * A[j * lda + j] + alpha * temp2;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsymm()\n");
  }
}

void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(SYRK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = (Trans == CblasConjTrans) ? CblasTrans : Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;

    if (Trans == CblasTrans || Trans == CblasConjTrans) {
      trans = CblasNoTrans;
    } else {
      trans = CblasTrans;
    }
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[i * lda + k] * A[j * lda + k];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[k * lda + i] * A[k * lda + j];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[i * lda + k] * A[j * lda + k];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += A[k * lda + i] * A[k * lda + j];
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsyrk()\n");
  }
}

void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const double alpha, const double *A, const int lda,
                  const double *B, const int ldb, const double beta,
                  double *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(SYR2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = (Trans == CblasConjTrans) ? CblasTrans : Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;

    if (Trans == CblasTrans || Trans == CblasConjTrans) {
      trans = CblasNoTrans;
    } else {
      trans = CblasTrans;
    }
  }

  /* form  C := beta*C */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += (A[i * lda + k] * B[j * ldb + k]
                   + B[i * ldb + k] * A[j * lda + k]);
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasTrans) {

    for (k = 0; k < K; k++) {
      for (i = 0; i < N; i++) {
        double temp1 = alpha * A[k * lda + i];
        double temp2 = alpha * B[k * ldb + i];
        for (j = i; j < N; j++) {
          C[i * lda + j] += temp1 * B[k * ldb + j] + temp2 * A[k * lda + j];
        }
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {


    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        double temp = 0.0;
        for (k = 0; k < K; k++) {
          temp += (A[i * lda + k] * B[j * ldb + k]
                   + B[i * ldb + k] * A[j * lda + k]);
        }
        C[i * ldc + j] += alpha * temp;
      }
    }

  } else if (uplo == CblasLower && trans == CblasTrans) {

    for (k = 0; k < K; k++) {
      for (i = 0; i < N; i++) {
        double temp1 = alpha * A[k * lda + i];
        double temp2 = alpha * B[k * ldb + i];
        for (j = 0; j <= i; j++) {
          C[i * lda + j] += temp1 * B[k * ldb + j] + temp2 * A[k * lda + j];
        }
      }
    }


  } else {
    fprintf(stderr, "unrecognized operation for cblas_dsyr2k()\n");
  }
}

void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  int side, uplo, trans;

  CHECK_ARGS12(TRMM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    side = Side;
    uplo = Uplo;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  } else {
    n1 = N;
    n2 = M;
    side = (Side == CblasLeft) ? CblasRight : CblasLeft;
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  }

  if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * TriU(A)*B */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        if (nonunit) {
          temp = A[i * lda + i] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = i + 1; k < n1; k++) {
          temp += A[lda * i + k] * B[k * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * (TriU(A))' *B */

    for (i = n1; i > 0 && i--;) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        for (k = 0; k < i; k++) {
          temp += A[lda * k + i] * B[k * ldb + j];
        }

        if (nonunit) {
          temp += A[i * lda + i] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * TriL(A)*B */


    for (i = n1; i > 0 && i--;) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        for (k = 0; k < i; k++) {
          temp += A[lda * i + k] * B[k * ldb + j];
        }

        if (nonunit) {
          temp += A[i * lda + i] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }



  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * TriL(A)' *B */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        if (nonunit) {
          temp = A[i * lda + i] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = i + 1; k < n1; k++) {
          temp += A[lda * k + i] * B[k * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * B * TriU(A) */

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {
        double temp = 0.0;

        for (k = 0; k < j; k++) {
          temp += A[lda * k + j] * B[i * ldb + k];
        }

        if (nonunit) {
          temp += A[j * lda + j] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * B * (TriU(A))' */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        if (nonunit) {
          temp = A[j * lda + j] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = j + 1; k < n2; k++) {
          temp += A[lda * j + k] * B[i * ldb + k];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha *B * TriL(A) */

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        double temp = 0.0;

        if (nonunit) {
          temp = A[j * lda + j] * B[i * ldb + j];
        } else {
          temp = B[i * ldb + j];
        }

        for (k = j + 1; k < n2; k++) {
          temp += A[lda * k + j] * B[i * ldb + k];
        }


        B[ldb * i + j] = alpha * temp;
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * B * TriL(A)' */

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {
        double temp = 0.0;

        for (k = 0; k < j; k++) {
          temp += A[lda * j + k] * B[i * ldb + k];
        }

        if (nonunit) {
          temp += A[j * lda + j] * B[i * ldb + j];
        } else {
          temp += B[i * ldb + j];
        }

        B[ldb * i + j] = alpha * temp;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_dtrmm()\n");
  }
}

void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
{
  int i, j, k;
  int n1, n2;
  const int nonunit = (Diag == CblasNonUnit);
  int side, uplo, trans;

  CHECK_ARGS12(TRSM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  if (Order == CblasRowMajor) {
    n1 = M;
    n2 = N;
    side = Side;
    uplo = Uplo;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  } else {
    n1 = N;
    n2 = M;
    side = (Side == CblasLeft) ? CblasRight : CblasLeft;
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (TransA == CblasConjTrans) ? CblasTrans : TransA;
  }

  if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * inv(TriU(A)) *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = n1; i > 0 && i--;) {
      if (nonunit) {
        double Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = 0; k < i; k++) {
        const double Aki = A[k * lda + i];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aki * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * inv(TriU(A))' *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      if (nonunit) {
        double Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = i + 1; k < n1; k++) {
        const double Aik = A[i * lda + k];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * inv(TriL(A))*B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      if (nonunit) {
        double Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = i + 1; k < n1; k++) {
        const double Aki = A[k * lda + i];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aki * B[ldb * i + j];
        }
      }
    }


  } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * TriL(A)' *B */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = n1; i > 0 && i--;) {
      if (nonunit) {
        double Aii = A[lda * i + i];
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] /= Aii;
        }
      }

      for (k = 0; k < i; k++) {
        const double Aik = A[i * lda + k];
        for (j = 0; j < n2; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

    /* form  B := alpha * B * inv(TriU(A)) */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        if (nonunit) {
          double Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          double Bij = B[ldb * i + j];
          for (k = j + 1; k < n2; k++) {
            B[ldb * i + k] -= A[j * lda + k] * Bij;
          }
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

    /* form  B := alpha * B * inv(TriU(A))' */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {

        if (nonunit) {
          double Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          double Bij = B[ldb * i + j];
          for (k = 0; k < j; k++) {
            B[ldb * i + k] -= A[k * lda + j] * Bij;
          }
        }
      }
    }


  } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

    /* form  B := alpha * B * inv(TriL(A)) */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = n2; j > 0 && j--;) {

        if (nonunit) {
          double Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          double Bij = B[ldb * i + j];
          for (k = 0; k < j; k++) {
            B[ldb * i + k] -= A[j * lda + k] * Bij;
          }
        }
      }
    }

  } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

    /* form  B := alpha * B * inv(TriL(A))' */
    if (alpha != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          B[ldb * i + j] *= alpha;
        }
      }
    }

    for (i = 0; i < n1; i++) {
      for (j = 0; j < n2; j++) {
        if (nonunit) {
          double Ajj = A[lda * j + j];
          B[ldb * i + j] /= Ajj;
        }

        {
          double Bij = B[ldb * i + j];
          for (k = j + 1; k < n2; k++) {
            B[ldb * i + k] -= A[k * lda + j] * Bij;
          }
        }
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_dtrsm()\n");
  }
}


void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int ldf, ldg;
  int conjF, conjG, TransF, TransG;
  const float *F, *G;

  CHECK_ARGS14(GEMM,Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      F = (const float *)A;
      ldf = lda;
      conjF = (TransA == CblasConjTrans) ? -1 : 1;
      TransF = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
      G = (const float *)B;
      ldg = ldb;
      conjG = (TransB == CblasConjTrans) ? -1 : 1;
      TransG = (TransB == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      F = (const float *)B;
      ldf = ldb;
      conjF = (TransB == CblasConjTrans) ? -1 : 1;
      TransF = (TransB == CblasNoTrans) ? CblasNoTrans : CblasTrans;
      G = (const float *)A;
      ldg = lda;
      conjG = (TransA == CblasConjTrans) ? -1 : 1;
      TransG = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    }

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_FLOAT(C, ldc * i + j) = 0.0;
          IMAG_FLOAT(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Cij_real = REAL_FLOAT(C, ldc * i + j);
          const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
          REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (TransF == CblasNoTrans && TransG == CblasNoTrans) {

      /* form  C := alpha*A*B + C */

      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const float Fik_real = CONST_REAL_FLOAT(F, ldf * i + k);
          const float Fik_imag = conjF * CONST_IMAG_FLOAT(F, ldf * i + k);
          const float temp_real = alpha_real * Fik_real - alpha_imag * Fik_imag;
          const float temp_imag = alpha_real * Fik_imag + alpha_imag * Fik_real;
          if (!(temp_real == 0.0 && temp_imag == 0.0)) {
            for (j = 0; j < n2; j++) {
              const float Gkj_real = CONST_REAL_FLOAT(G, ldg * k + j);
              const float Gkj_imag = conjG * CONST_IMAG_FLOAT(G, ldg * k + j);
              REAL_FLOAT(C, ldc * i + j) += temp_real * Gkj_real - temp_imag * Gkj_imag;
              IMAG_FLOAT(C, ldc * i + j) += temp_real * Gkj_imag + temp_imag * Gkj_real;
            }
          }
        }
      }

    } else if (TransF == CblasNoTrans && TransG == CblasTrans) {

      /* form  C := alpha*A*B' + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Fik_real = CONST_REAL_FLOAT(F, ldf * i + k);
            const float Fik_imag = conjF * CONST_IMAG_FLOAT(F, ldf * i + k);
            const float Gjk_real = CONST_REAL_FLOAT(G, ldg * j + k);
            const float Gjk_imag = conjG * CONST_IMAG_FLOAT(G, ldg * j + k);
            temp_real += Fik_real * Gjk_real - Fik_imag * Gjk_imag;
            temp_imag += Fik_real * Gjk_imag + Fik_imag * Gjk_real;
          }
          REAL_FLOAT(C, ldc * i + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, ldc * i + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (TransF == CblasTrans && TransG == CblasNoTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const float Fki_real = CONST_REAL_FLOAT(F, ldf * k + i);
          const float Fki_imag = conjF * CONST_IMAG_FLOAT(F, ldf * k + i);
          const float temp_real = alpha_real * Fki_real - alpha_imag * Fki_imag;
          const float temp_imag = alpha_real * Fki_imag + alpha_imag * Fki_real;
          if (!(temp_real == 0.0 && temp_imag == 0.0)) {
            for (j = 0; j < n2; j++) {
              const float Gkj_real = CONST_REAL_FLOAT(G, ldg * k + j);
              const float Gkj_imag = conjG * CONST_IMAG_FLOAT(G, ldg * k + j);
              REAL_FLOAT(C, ldc * i + j) += temp_real * Gkj_real - temp_imag * Gkj_imag;
              IMAG_FLOAT(C, ldc * i + j) += temp_real * Gkj_imag + temp_imag * Gkj_real;
            }
          }
        }
      }

    } else if (TransF == CblasTrans && TransG == CblasTrans) {

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Fki_real = CONST_REAL_FLOAT(F, ldf * k + i);
            const float Fki_imag = conjF * CONST_IMAG_FLOAT(F, ldf * k + i);
            const float Gjk_real = CONST_REAL_FLOAT(G, ldg * j + k);
            const float Gjk_imag = conjG * CONST_IMAG_FLOAT(G, ldg * j + k);

            temp_real += Fki_real * Gjk_real - Fki_imag * Gjk_imag;
            temp_imag += Fki_real * Gjk_imag + Fki_imag * Gjk_real;
          }
          REAL_FLOAT(C, ldc * i + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, ldc * i + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_cgemm()\n");
    }
  }
}

void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int uplo, side;

  CHECK_ARGS13(SYMM,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);
    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
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
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_FLOAT(C, ldc * i + j) = 0.0;
          IMAG_FLOAT(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Cij_real = REAL_FLOAT(C, ldc * i + j);
          const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
          REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (side == CblasLeft && uplo == CblasUpper) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = CONST_IMAG_FLOAT(A, i * lda + i);
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Aii_real - temp1_imag * Aii_imag;
            IMAG_FLOAT(C, i * ldc + j) += temp1_real * Aii_imag + temp1_imag * Aii_real;
          }
          for (k = i + 1; k < n1; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = CONST_REAL_FLOAT(B, ldb * k + j);
            const float Bkj_imag = CONST_IMAG_FLOAT(B, ldb * k + j);
            REAL_FLOAT(C, k * ldc + j) += Aik_real * temp1_real - Aik_imag * temp1_imag;
            IMAG_FLOAT(C, k * ldc + j) += Aik_real * temp1_imag + Aik_imag * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          for (k = 0; k < i; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = CONST_REAL_FLOAT(B, ldb * k + j);
            const float Bkj_imag = CONST_IMAG_FLOAT(B, ldb * k + j);
            REAL_FLOAT(C, k * ldc + j) += Aik_real * temp1_real - Aik_imag * temp1_imag;
            IMAG_FLOAT(C, k * ldc + j) += Aik_real * temp1_imag + Aik_imag * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = CONST_IMAG_FLOAT(A, i * lda + i);
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Aii_real - temp1_imag * Aii_imag;
            IMAG_FLOAT(C, i * ldc + j) += temp1_real * Aii_imag + temp1_imag * Aii_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = CONST_IMAG_FLOAT(A, j * lda + j);
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Ajj_real - temp1_imag * Ajj_imag;
            IMAG_FLOAT(C, i * ldc + j) += temp1_real * Ajj_imag + temp1_imag * Ajj_real;
          }
          for (k = j + 1; k < n2; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, ldb * i + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, ldb * i + k);
            REAL_FLOAT(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_FLOAT(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * Ajk_imag;
            temp2_imag += Bik_real * Ajk_imag + Bik_imag * Ajk_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          for (k = 0; k < j; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, ldb * i + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, ldb * i + k);
            REAL_FLOAT(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_FLOAT(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * Ajk_imag;
            temp2_imag += Bik_real * Ajk_imag + Bik_imag * Ajk_real;
          }
          {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = CONST_IMAG_FLOAT(A, j * lda + j);
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Ajj_real - temp1_imag * Ajj_imag;
            IMAG_FLOAT(C, i * ldc + j) += temp1_real * Ajj_imag + temp1_imag * Ajj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_csymm()\n");
    }
  }
}

void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *beta, void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(SYRK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      /* FIXME: original blas does not make distinction between Trans and ConjTrans?? */
      trans = (Trans == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    }

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            const float Cij_real = REAL_FLOAT(C, ldc * i + j);
            const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
            REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            const float Cij_real = REAL_FLOAT(C, ldc * i + j);
            const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
            REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
            temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
            const float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
            const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            const float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
            temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
            temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasTrans) {

      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
            const float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
            const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            const float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
            temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_csyrk()\n");
    }
  }
}

void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const void *beta,
                  void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(SYR2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);
    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      trans = Trans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    }

    /* form  C := beta*C */

    if (beta_real == 0.0 && beta_imag == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            const float Cij_real = REAL_FLOAT(C, ldc * i + j);
            const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
            REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            const float Cij_real = REAL_FLOAT(C, ldc * i + j);
            const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
            REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      }
    }


    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bjk_real = CONST_REAL_FLOAT(B, j * ldb + k);
            const float Bjk_imag = CONST_IMAG_FLOAT(B, j * ldb + k);
            temp_real += ((Aik_real * Bjk_real - Aik_imag * Bjk_imag)
                          + (Bik_real * Ajk_real - Bik_imag * Ajk_imag));
            temp_imag += ((Aik_real * Bjk_imag + Aik_imag * Bjk_real)
                          + (Bik_real * Ajk_imag + Bik_imag * Ajk_real));
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
          float Bki_real = CONST_REAL_FLOAT(B, k * ldb + i);
          float Bki_imag = CONST_IMAG_FLOAT(B, k * ldb + i);
          float temp1_real = alpha_real * Aki_real - alpha_imag * Aki_imag;
          float temp1_imag = alpha_real * Aki_imag + alpha_imag * Aki_real;
          float temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          float temp2_imag = alpha_real * Bki_imag + alpha_imag * Bki_real;
          for (j = i; j < N; j++) {
            float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            float Bkj_real = CONST_REAL_FLOAT(B, k * ldb + j);
            float Bkj_imag = CONST_IMAG_FLOAT(B, k * ldb + j);
            REAL_FLOAT(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_FLOAT(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {


      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bjk_real = CONST_REAL_FLOAT(B, j * ldb + k);
            const float Bjk_imag = CONST_IMAG_FLOAT(B, j * ldb + k);
            temp_real += ((Aik_real * Bjk_real - Aik_imag * Bjk_imag)
                          + (Bik_real * Ajk_real - Bik_imag * Ajk_imag));
            temp_imag += ((Aik_real * Bjk_imag + Aik_imag * Bjk_real)
                          + (Bik_real * Ajk_imag + Bik_imag * Ajk_real));
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
          float Bki_real = CONST_REAL_FLOAT(B, k * ldb + i);
          float Bki_imag = CONST_IMAG_FLOAT(B, k * ldb + i);
          float temp1_real = alpha_real * Aki_real - alpha_imag * Aki_imag;
          float temp1_imag = alpha_real * Aki_imag + alpha_imag * Aki_real;
          float temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          float temp2_imag = alpha_real * Bki_imag + alpha_imag * Bki_real;
          for (j = 0; j <= i; j++) {
            float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            float Bkj_real = CONST_REAL_FLOAT(B, k * ldb + j);
            float Bkj_imag = CONST_IMAG_FLOAT(B, k * ldb + j);
            REAL_FLOAT(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_FLOAT(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }


    } else {
      fprintf(stderr, "unrecognized operation for cblas_csyr2k()\n");
    }
  }
}

void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  int side, uplo, trans;

  CHECK_ARGS12(TRMM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      side = Side;
      uplo = Uplo;
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      side = (Side == CblasLeft) ? CblasRight : CblasLeft;        /* exchanged */
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;      /* exchanged */
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;       /* same */
    }

    if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * TriU(A)*B */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          if (nonunit) {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = conj * CONST_IMAG_FLOAT(A, i * lda + i);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real = Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag = Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real = REAL_FLOAT(B, i * ldb + j);
            temp_imag = IMAG_FLOAT(B, i * ldb + j);
          }

          for (k = i + 1; k < n1; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = conj * CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = REAL_FLOAT(B, k * ldb + j);
            const float Bkj_imag = IMAG_FLOAT(B, k * ldb + j);
            temp_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * (TriU(A))' *B */

      for (i = n1; i > 0 && i--;) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          for (k = 0; k < i; k++) {
            const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
            const float Aki_imag = conj * CONST_IMAG_FLOAT(A, k * lda + i);
            const float Bkj_real = REAL_FLOAT(B, k * ldb + j);
            const float Bkj_imag = IMAG_FLOAT(B, k * ldb + j);
            temp_real += Aki_real * Bkj_real - Aki_imag * Bkj_imag;
            temp_imag += Aki_real * Bkj_imag + Aki_imag * Bkj_real;
          }

          if (nonunit) {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = conj * CONST_IMAG_FLOAT(A, i * lda + i);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real += Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag += Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real += REAL_FLOAT(B, i * ldb + j);
            temp_imag += IMAG_FLOAT(B, i * ldb + j);
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * TriL(A)*B */


      for (i = n1; i > 0 && i--;) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          for (k = 0; k < i; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = conj * CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = REAL_FLOAT(B, k * ldb + j);
            const float Bkj_imag = IMAG_FLOAT(B, k * ldb + j);
            temp_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }

          if (nonunit) {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = conj * CONST_IMAG_FLOAT(A, i * lda + i);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real += Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag += Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real += REAL_FLOAT(B, i * ldb + j);
            temp_imag += IMAG_FLOAT(B, i * ldb + j);
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }



    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * TriL(A)' *B */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          if (nonunit) {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            const float Aii_imag = conj * CONST_IMAG_FLOAT(A, i * lda + i);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real = Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag = Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real = REAL_FLOAT(B, i * ldb + j);
            temp_imag = IMAG_FLOAT(B, i * ldb + j);
          }

          for (k = i + 1; k < n1; k++) {
            const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
            const float Aki_imag = conj * CONST_IMAG_FLOAT(A, k * lda + i);
            const float Bkj_real = REAL_FLOAT(B, k * ldb + j);
            const float Bkj_imag = IMAG_FLOAT(B, k * ldb + j);
            temp_real += Aki_real * Bkj_real - Aki_imag * Bkj_imag;
            temp_imag += Aki_real * Bkj_imag + Aki_imag * Bkj_real;
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * B * TriU(A) */

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          for (k = 0; k < j; k++) {
            const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            const float Akj_imag = conj * CONST_IMAG_FLOAT(A, k * lda + j);
            const float Bik_real = REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = IMAG_FLOAT(B, i * ldb + k);
            temp_real += Akj_real * Bik_real - Akj_imag * Bik_imag;
            temp_imag += Akj_real * Bik_imag + Akj_imag * Bik_real;
          }

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, j * lda + j);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real += Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag += Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real += REAL_FLOAT(B, i * ldb + j);
            temp_imag += IMAG_FLOAT(B, i * ldb + j);
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * B * (TriU(A))' */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, j * lda + j);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real = Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag = Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real = REAL_FLOAT(B, i * ldb + j);
            temp_imag = IMAG_FLOAT(B, i * ldb + j);
          }

          for (k = j + 1; k < n2; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = conj * CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = IMAG_FLOAT(B, i * ldb + k);
            temp_real += Ajk_real * Bik_real - Ajk_imag * Bik_imag;
            temp_imag += Ajk_real * Bik_imag + Ajk_imag * Bik_real;
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha *B * TriL(A) */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, j * lda + j);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real = Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag = Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real = REAL_FLOAT(B, i * ldb + j);
            temp_imag = IMAG_FLOAT(B, i * ldb + j);
          }

          for (k = j + 1; k < n2; k++) {
            const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            const float Akj_imag = conj * CONST_IMAG_FLOAT(A, k * lda + j);
            const float Bik_real = REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = IMAG_FLOAT(B, i * ldb + k);
            temp_real += Akj_real * Bik_real - Akj_imag * Bik_imag;
            temp_imag += Akj_real * Bik_imag + Akj_imag * Bik_real;
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * B * TriL(A)' */

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {
          float temp_real = 0.0;
          float temp_imag = 0.0;

          for (k = 0; k < j; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = conj * CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = IMAG_FLOAT(B, i * ldb + k);
            temp_real += Ajk_real * Bik_real - Ajk_imag * Bik_imag;
            temp_imag += Ajk_real * Bik_imag + Ajk_imag * Bik_real;
          }

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, j * lda + j);
            const float Bij_real = REAL_FLOAT(B, i * ldb + j);
            const float Bij_imag = IMAG_FLOAT(B, i * ldb + j);
            temp_real += Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag += Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real += REAL_FLOAT(B, i * ldb + j);
            temp_imag += IMAG_FLOAT(B, i * ldb + j);
          }

          REAL_FLOAT(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_FLOAT(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_ctrmm()\n");
    }
  }
}


void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  int side, uplo, trans;

  CHECK_ARGS12(TRSM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      side = Side;
      uplo = Uplo;
      trans = TransA;
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      side = (Side == CblasLeft) ? CblasRight : CblasLeft;        /* exchanged */
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;      /* exchanged */
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;       /* same */
    }

    if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * inv(TriU(A)) *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = n1; i > 0 && i--;) {
        if (nonunit) {
          const float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
          const float Aii_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
          const float s = xhypot(Aii_real, Aii_imag);
          const float a_real = Aii_real / s;
          const float a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = 0; k < i; k++) {
          const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          const float Aki_imag = conj * CONST_IMAG_FLOAT(A, k * lda + i);
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * k + j) -= Aki_real * Bij_real - Aki_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * k + j) -= Aki_real * Bij_imag + Aki_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * inv(TriU(A))' *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {

        if (nonunit) {
          const float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
          const float Aii_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
          const float s = xhypot(Aii_real, Aii_imag);
          const float a_real = Aii_real / s;
          const float a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = i + 1; k < n1; k++) {
          const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
          const float Aik_imag = conj * CONST_IMAG_FLOAT(A, i * lda + k);
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * k + j) -= Aik_real * Bij_real - Aik_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * k + j) -= Aik_real * Bij_imag + Aik_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * inv(TriL(A))*B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {

        if (nonunit) {
          const float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
          const float Aii_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
          const float s = xhypot(Aii_real, Aii_imag);
          const float a_real = Aii_real / s;
          const float a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = i + 1; k < n1; k++) {
          const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          const float Aki_imag = conj * CONST_IMAG_FLOAT(A, k * lda + i);
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * k + j) -= Aki_real * Bij_real - Aki_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * k + j) -= Aki_real * Bij_imag + Aki_imag * Bij_real;
          }
        }
      }


    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * TriL(A)' *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = n1; i > 0 && i--;) {
        if (nonunit) {
          const float Aii_real = CONST_REAL_FLOAT(A, lda * i + i);
          const float Aii_imag = conj * CONST_IMAG_FLOAT(A, lda * i + i);
          const float s = xhypot(Aii_real, Aii_imag);
          const float a_real = Aii_real / s;
          const float a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = 0; k < i; k++) {
          const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
          const float Aik_imag = conj * CONST_IMAG_FLOAT(A, i * lda + k);
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * k + j) -= Aik_real * Bij_real - Aik_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * k + j) -= Aik_real * Bij_imag + Aik_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * B * inv(TriU(A)) */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, lda * j + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, lda * j + j);
            const float s = xhypot(Ajj_real, Ajj_imag);
            const float a_real = Ajj_real / s;
            const float a_imag = Ajj_imag / s;
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            for (k = j + 1; k < n2; k++) {
              const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
              const float Ajk_imag = conj * CONST_IMAG_FLOAT(A, j * lda + k);
              REAL_FLOAT(B, ldb * i + k) -= Ajk_real * Bij_real - Ajk_imag * Bij_imag;
              IMAG_FLOAT(B, ldb * i + k) -= Ajk_real * Bij_imag + Ajk_imag * Bij_real;
            }
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * B * inv(TriU(A))' */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, lda * j + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, lda * j + j);
            const float s = xhypot(Ajj_real, Ajj_imag);
            const float a_real = Ajj_real / s;
            const float a_imag = Ajj_imag / s;
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            for (k = 0; k < j; k++) {
              const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
              const float Akj_imag = conj * CONST_IMAG_FLOAT(A, k * lda + j);
              REAL_FLOAT(B, ldb * i + k) -= Akj_real * Bij_real - Akj_imag * Bij_imag;
              IMAG_FLOAT(B, ldb * i + k) -= Akj_real * Bij_imag + Akj_imag * Bij_real;
            }
          }
        }
      }


    } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * B * inv(TriL(A)) */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {

          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, lda * j + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, lda * j + j);
            const float s = xhypot(Ajj_real, Ajj_imag);
            const float a_real = Ajj_real / s;
            const float a_imag = Ajj_imag / s;
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            for (k = 0; k < j; k++) {
              const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
              const float Ajk_imag = conj * CONST_IMAG_FLOAT(A, j * lda + k);
              REAL_FLOAT(B, ldb * i + k) -= Ajk_real * Bij_real - Ajk_imag * Bij_imag;
              IMAG_FLOAT(B, ldb * i + k) -= Ajk_real * Bij_imag + Ajk_imag * Bij_real;
            }
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * B * inv(TriL(A))' */


      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_FLOAT(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          if (nonunit) {
            const float Ajj_real = CONST_REAL_FLOAT(A, lda * j + j);
            const float Ajj_imag = conj * CONST_IMAG_FLOAT(A, lda * j + j);
            const float s = xhypot(Ajj_real, Ajj_imag);
            const float a_real = Ajj_real / s;
            const float a_imag = Ajj_imag / s;
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);
            REAL_FLOAT(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_FLOAT(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const float Bij_real = REAL_FLOAT(B, ldb * i + j);
            const float Bij_imag = IMAG_FLOAT(B, ldb * i + j);

            for (k = j + 1; k < n2; k++) {
              const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
              const float Akj_imag = conj * CONST_IMAG_FLOAT(A, k * lda + j);
              REAL_FLOAT(B, ldb * i + k) -= Akj_real * Bij_real - Akj_imag * Bij_imag;
              IMAG_FLOAT(B, ldb * i + k) -= Akj_real * Bij_imag + Akj_imag * Bij_real;
            }
          }
        }
      }


    } else {
      fprintf(stderr, "unrecognized operation for cblas_ctrsm()\n");
    }
  }
}

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int ldf, ldg;
  int conjF, conjG, TransF, TransG;
  const double *F, *G;

  CHECK_ARGS14(GEMM,Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      F = (const double *)A;
      ldf = lda;
      conjF = (TransA == CblasConjTrans) ? -1 : 1;
      TransF = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
      G = (const double *)B;
      ldg = ldb;
      conjG = (TransB == CblasConjTrans) ? -1 : 1;
      TransG = (TransB == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      F = (const double *)B;
      ldf = ldb;
      conjF = (TransB == CblasConjTrans) ? -1 : 1;
      TransF = (TransB == CblasNoTrans) ? CblasNoTrans : CblasTrans;
      G = (const double *)A;
      ldg = lda;
      conjG = (TransA == CblasConjTrans) ? -1 : 1;
      TransG = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    }

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_DOUBLE(C, ldc * i + j) = 0.0;
          IMAG_DOUBLE(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
          const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
          REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (TransF == CblasNoTrans && TransG == CblasNoTrans) {

      /* form  C := alpha*A*B + C */

      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const double Fik_real = CONST_REAL_DOUBLE(F, ldf * i + k);
          const double Fik_imag = conjF * CONST_IMAG_DOUBLE(F, ldf * i + k);
          const double temp_real = alpha_real * Fik_real - alpha_imag * Fik_imag;
          const double temp_imag = alpha_real * Fik_imag + alpha_imag * Fik_real;
          if (!(temp_real == 0.0 && temp_imag == 0.0)) {
            for (j = 0; j < n2; j++) {
              const double Gkj_real = CONST_REAL_DOUBLE(G, ldg * k + j);
              const double Gkj_imag = conjG * CONST_IMAG_DOUBLE(G, ldg * k + j);
              REAL_DOUBLE(C, ldc * i + j) += temp_real * Gkj_real - temp_imag * Gkj_imag;
              IMAG_DOUBLE(C, ldc * i + j) += temp_real * Gkj_imag + temp_imag * Gkj_real;
            }
          }
        }
      }

    } else if (TransF == CblasNoTrans && TransG == CblasTrans) {

      /* form  C := alpha*A*B' + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Fik_real = CONST_REAL_DOUBLE(F, ldf * i + k);
            const double Fik_imag = conjF * CONST_IMAG_DOUBLE(F, ldf * i + k);
            const double Gjk_real = CONST_REAL_DOUBLE(G, ldg * j + k);
            const double Gjk_imag = conjG * CONST_IMAG_DOUBLE(G, ldg * j + k);
            temp_real += Fik_real * Gjk_real - Fik_imag * Gjk_imag;
            temp_imag += Fik_real * Gjk_imag + Fik_imag * Gjk_real;
          }
          REAL_DOUBLE(C, ldc * i + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, ldc * i + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (TransF == CblasTrans && TransG == CblasNoTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const double Fki_real = CONST_REAL_DOUBLE(F, ldf * k + i);
          const double Fki_imag = conjF * CONST_IMAG_DOUBLE(F, ldf * k + i);
          const double temp_real = alpha_real * Fki_real - alpha_imag * Fki_imag;
          const double temp_imag = alpha_real * Fki_imag + alpha_imag * Fki_real;
          if (!(temp_real == 0.0 && temp_imag == 0.0)) {
            for (j = 0; j < n2; j++) {
              const double Gkj_real = CONST_REAL_DOUBLE(G, ldg * k + j);
              const double Gkj_imag = conjG * CONST_IMAG_DOUBLE(G, ldg * k + j);
              REAL_DOUBLE(C, ldc * i + j) += temp_real * Gkj_real - temp_imag * Gkj_imag;
              IMAG_DOUBLE(C, ldc * i + j) += temp_real * Gkj_imag + temp_imag * Gkj_real;
            }
          }
        }
      }

    } else if (TransF == CblasTrans && TransG == CblasTrans) {

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Fki_real = CONST_REAL_DOUBLE(F, ldf * k + i);
            const double Fki_imag = conjF * CONST_IMAG_DOUBLE(F, ldf * k + i);
            const double Gjk_real = CONST_REAL_DOUBLE(G, ldg * j + k);
            const double Gjk_imag = conjG * CONST_IMAG_DOUBLE(G, ldg * j + k);

            temp_real += Fki_real * Gjk_real - Fki_imag * Gjk_imag;
            temp_imag += Fki_real * Gjk_imag + Fki_imag * Gjk_real;
          }
          REAL_DOUBLE(C, ldc * i + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, ldc * i + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zgemm()\n");
    }
  }
}


void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int uplo, side;

  CHECK_ARGS13(SYMM,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);
    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
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
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_DOUBLE(C, ldc * i + j) = 0.0;
          IMAG_DOUBLE(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
          const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
          REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (side == CblasLeft && uplo == CblasUpper) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = CONST_IMAG_DOUBLE(A, i * lda + i);
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Aii_real - temp1_imag * Aii_imag;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_real * Aii_imag + temp1_imag * Aii_real;
          }
          for (k = i + 1; k < n1; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = CONST_REAL_DOUBLE(B, ldb * k + j);
            const double Bkj_imag = CONST_IMAG_DOUBLE(B, ldb * k + j);
            REAL_DOUBLE(C, k * ldc + j) += Aik_real * temp1_real - Aik_imag * temp1_imag;
            IMAG_DOUBLE(C, k * ldc + j) += Aik_real * temp1_imag + Aik_imag * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          for (k = 0; k < i; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = CONST_REAL_DOUBLE(B, ldb * k + j);
            const double Bkj_imag = CONST_IMAG_DOUBLE(B, ldb * k + j);
            REAL_DOUBLE(C, k * ldc + j) += Aik_real * temp1_real - Aik_imag * temp1_imag;
            IMAG_DOUBLE(C, k * ldc + j) += Aik_real * temp1_imag + Aik_imag * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = CONST_IMAG_DOUBLE(A, i * lda + i);
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Aii_real - temp1_imag * Aii_imag;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_real * Aii_imag + temp1_imag * Aii_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = CONST_IMAG_DOUBLE(A, j * lda + j);
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_real - temp1_imag * Ajj_imag;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_imag + temp1_imag * Ajj_real;
          }
          for (k = j + 1; k < n2; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, ldb * i + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, ldb * i + k);
            REAL_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * Ajk_imag;
            temp2_imag += Bik_real * Ajk_imag + Bik_imag * Ajk_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          for (k = 0; k < j; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, ldb * i + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, ldb * i + k);
            REAL_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * Ajk_imag;
            temp2_imag += Bik_real * Ajk_imag + Bik_imag * Ajk_real;
          }
          {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = CONST_IMAG_DOUBLE(A, j * lda + j);
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_real - temp1_imag * Ajj_imag;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_imag + temp1_imag * Ajj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zsymm()\n");
    }
  }
}

void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *beta, void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(SYRK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      /* FIXME: original blas does not make distinction between Trans and ConjTrans?? */
      trans = (Trans == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    }

    /* form  y := beta*y */
    if (beta_real == 0.0 && beta_imag == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
            const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
            REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
            const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
            REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
            temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
            const double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
            const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            const double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
            temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
            temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasTrans) {

      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
            const double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
            const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            const double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
            temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zsyrk()\n");
    }
  }
}


void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const void *beta,
                  void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(SYR2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);
    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      trans = Trans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    }

    /* form  C := beta*C */

    if (beta_real == 0.0 && beta_imag == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
            const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
            REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
            const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
            REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
            IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
          }
        }
      }
    }


    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bjk_real = CONST_REAL_DOUBLE(B, j * ldb + k);
            const double Bjk_imag = CONST_IMAG_DOUBLE(B, j * ldb + k);
            temp_real += ((Aik_real * Bjk_real - Aik_imag * Bjk_imag)
                          + (Bik_real * Ajk_real - Bik_imag * Ajk_imag));
            temp_imag += ((Aik_real * Bjk_imag + Aik_imag * Bjk_real)
                          + (Bik_real * Ajk_imag + Bik_imag * Ajk_real));
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
          double Bki_real = CONST_REAL_DOUBLE(B, k * ldb + i);
          double Bki_imag = CONST_IMAG_DOUBLE(B, k * ldb + i);
          double temp1_real = alpha_real * Aki_real - alpha_imag * Aki_imag;
          double temp1_imag = alpha_real * Aki_imag + alpha_imag * Aki_real;
          double temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          double temp2_imag = alpha_real * Bki_imag + alpha_imag * Bki_real;
          for (j = i; j < N; j++) {
            double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            double Bkj_real = CONST_REAL_DOUBLE(B, k * ldb + j);
            double Bkj_imag = CONST_IMAG_DOUBLE(B, k * ldb + j);
            REAL_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {


      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bjk_real = CONST_REAL_DOUBLE(B, j * ldb + k);
            const double Bjk_imag = CONST_IMAG_DOUBLE(B, j * ldb + k);
            temp_real += ((Aik_real * Bjk_real - Aik_imag * Bjk_imag)
                          + (Bik_real * Ajk_real - Bik_imag * Ajk_imag));
            temp_imag += ((Aik_real * Bjk_imag + Aik_imag * Bjk_real)
                          + (Bik_real * Ajk_imag + Bik_imag * Ajk_real));
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (uplo == CblasLower && trans == CblasTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
          double Bki_real = CONST_REAL_DOUBLE(B, k * ldb + i);
          double Bki_imag = CONST_IMAG_DOUBLE(B, k * ldb + i);
          double temp1_real = alpha_real * Aki_real - alpha_imag * Aki_imag;
          double temp1_imag = alpha_real * Aki_imag + alpha_imag * Aki_real;
          double temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          double temp2_imag = alpha_real * Bki_imag + alpha_imag * Bki_real;
          for (j = 0; j <= i; j++) {
            double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            double Bkj_real = CONST_REAL_DOUBLE(B, k * ldb + j);
            double Bkj_imag = CONST_IMAG_DOUBLE(B, k * ldb + j);
            REAL_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }


    } else {
      fprintf(stderr, "unrecognized operation for cblas_zsyr2k()\n");
    }
  }
}

void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  int side, uplo, trans;

  CHECK_ARGS12(TRMM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      side = Side;
      uplo = Uplo;
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      side = (Side == CblasLeft) ? CblasRight : CblasLeft;        /* exchanged */
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;      /* exchanged */
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;       /* same */
    }

    if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * TriU(A)*B */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          if (nonunit) {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + i);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real = Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag = Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real = REAL_DOUBLE(B, i * ldb + j);
            temp_imag = IMAG_DOUBLE(B, i * ldb + j);
          }

          for (k = i + 1; k < n1; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = REAL_DOUBLE(B, k * ldb + j);
            const double Bkj_imag = IMAG_DOUBLE(B, k * ldb + j);
            temp_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * (TriU(A))' *B */

      for (i = n1; i > 0 && i--;) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          for (k = 0; k < i; k++) {
            const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
            const double Aki_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + i);
            const double Bkj_real = REAL_DOUBLE(B, k * ldb + j);
            const double Bkj_imag = IMAG_DOUBLE(B, k * ldb + j);
            temp_real += Aki_real * Bkj_real - Aki_imag * Bkj_imag;
            temp_imag += Aki_real * Bkj_imag + Aki_imag * Bkj_real;
          }

          if (nonunit) {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + i);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real += Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag += Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real += REAL_DOUBLE(B, i * ldb + j);
            temp_imag += IMAG_DOUBLE(B, i * ldb + j);
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * TriL(A)*B */


      for (i = n1; i > 0 && i--;) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          for (k = 0; k < i; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = REAL_DOUBLE(B, k * ldb + j);
            const double Bkj_imag = IMAG_DOUBLE(B, k * ldb + j);
            temp_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }

          if (nonunit) {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + i);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real += Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag += Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real += REAL_DOUBLE(B, i * ldb + j);
            temp_imag += IMAG_DOUBLE(B, i * ldb + j);
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }



    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * TriL(A)' *B */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          if (nonunit) {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + i);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real = Aii_real * Bij_real - Aii_imag * Bij_imag;
            temp_imag = Aii_real * Bij_imag + Aii_imag * Bij_real;
          } else {
            temp_real = REAL_DOUBLE(B, i * ldb + j);
            temp_imag = IMAG_DOUBLE(B, i * ldb + j);
          }

          for (k = i + 1; k < n1; k++) {
            const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
            const double Aki_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + i);
            const double Bkj_real = REAL_DOUBLE(B, k * ldb + j);
            const double Bkj_imag = IMAG_DOUBLE(B, k * ldb + j);
            temp_real += Aki_real * Bkj_real - Aki_imag * Bkj_imag;
            temp_imag += Aki_real * Bkj_imag + Aki_imag * Bkj_real;
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * B * TriU(A) */

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          for (k = 0; k < j; k++) {
            const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            const double Akj_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + j);
            const double Bik_real = REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = IMAG_DOUBLE(B, i * ldb + k);
            temp_real += Akj_real * Bik_real - Akj_imag * Bik_imag;
            temp_imag += Akj_real * Bik_imag + Akj_imag * Bik_real;
          }

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + j);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real += Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag += Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real += REAL_DOUBLE(B, i * ldb + j);
            temp_imag += IMAG_DOUBLE(B, i * ldb + j);
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * B * (TriU(A))' */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + j);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real = Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag = Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real = REAL_DOUBLE(B, i * ldb + j);
            temp_imag = IMAG_DOUBLE(B, i * ldb + j);
          }

          for (k = j + 1; k < n2; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = IMAG_DOUBLE(B, i * ldb + k);
            temp_real += Ajk_real * Bik_real - Ajk_imag * Bik_imag;
            temp_imag += Ajk_real * Bik_imag + Ajk_imag * Bik_real;
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha *B * TriL(A) */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + j);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real = Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag = Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real = REAL_DOUBLE(B, i * ldb + j);
            temp_imag = IMAG_DOUBLE(B, i * ldb + j);
          }

          for (k = j + 1; k < n2; k++) {
            const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            const double Akj_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + j);
            const double Bik_real = REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = IMAG_DOUBLE(B, i * ldb + k);
            temp_real += Akj_real * Bik_real - Akj_imag * Bik_imag;
            temp_imag += Akj_real * Bik_imag + Akj_imag * Bik_real;
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * B * TriL(A)' */

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {
          double temp_real = 0.0;
          double temp_imag = 0.0;

          for (k = 0; k < j; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = IMAG_DOUBLE(B, i * ldb + k);
            temp_real += Ajk_real * Bik_real - Ajk_imag * Bik_imag;
            temp_imag += Ajk_real * Bik_imag + Ajk_imag * Bik_real;
          }

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + j);
            const double Bij_real = REAL_DOUBLE(B, i * ldb + j);
            const double Bij_imag = IMAG_DOUBLE(B, i * ldb + j);
            temp_real += Ajj_real * Bij_real - Ajj_imag * Bij_imag;
            temp_imag += Ajj_real * Bij_imag + Ajj_imag * Bij_real;
          } else {
            temp_real += REAL_DOUBLE(B, i * ldb + j);
            temp_imag += IMAG_DOUBLE(B, i * ldb + j);
          }

          REAL_DOUBLE(B, ldb * i + j) = alpha_real * temp_real - alpha_imag * temp_imag;
          IMAG_DOUBLE(B, ldb * i + j) = alpha_real * temp_imag + alpha_imag * temp_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_ztrmm()\n");
    }
  }
}


void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb)
{
  int i, j, k;
  int n1, n2;

  const int nonunit = (Diag == CblasNonUnit);
  const int conj = (TransA == CblasConjTrans) ? -1 : 1;
  int side, uplo, trans;

  CHECK_ARGS12(TRSM,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      side = Side;
      uplo = Uplo;
      trans = TransA;
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;
    } else {
      n1 = N;
      n2 = M;
      side = (Side == CblasLeft) ? CblasRight : CblasLeft;        /* exchanged */
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;      /* exchanged */
      trans = (TransA == CblasNoTrans) ? CblasNoTrans : CblasTrans;       /* same */
    }

    if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * inv(TriU(A)) *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = n1; i > 0 && i--;) {
        if (nonunit) {
          const double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
          const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
          const double s = xhypot(Aii_real, Aii_imag);
          const double a_real = Aii_real / s;
          const double a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = 0; k < i; k++) {
          const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          const double Aki_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + i);
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * k + j) -= Aki_real * Bij_real - Aki_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * k + j) -= Aki_real * Bij_imag + Aki_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * inv(TriU(A))' *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {

        if (nonunit) {
          const double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
          const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
          const double s = xhypot(Aii_real, Aii_imag);
          const double a_real = Aii_real / s;
          const double a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = i + 1; k < n1; k++) {
          const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
          const double Aik_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + k);
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * k + j) -= Aik_real * Bij_real - Aik_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * k + j) -= Aik_real * Bij_imag + Aik_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * inv(TriL(A))*B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {

        if (nonunit) {
          const double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
          const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
          const double s = xhypot(Aii_real, Aii_imag);
          const double a_real = Aii_real / s;
          const double a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = i + 1; k < n1; k++) {
          const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          const double Aki_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + i);
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * k + j) -= Aki_real * Bij_real - Aki_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * k + j) -= Aki_real * Bij_imag + Aki_imag * Bij_real;
          }
        }
      }


    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * TriL(A)' *B */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = n1; i > 0 && i--;) {
        if (nonunit) {
          const double Aii_real = CONST_REAL_DOUBLE(A, lda * i + i);
          const double Aii_imag = conj * CONST_IMAG_DOUBLE(A, lda * i + i);
          const double s = xhypot(Aii_real, Aii_imag);
          const double a_real = Aii_real / s;
          const double a_imag = Aii_imag / s;

          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }
        }

        for (k = 0; k < i; k++) {
          const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
          const double Aik_imag = conj * CONST_IMAG_DOUBLE(A, i * lda + k);
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * k + j) -= Aik_real * Bij_real - Aik_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * k + j) -= Aik_real * Bij_imag + Aik_imag * Bij_real;
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasNoTrans) {

      /* form  B := alpha * B * inv(TriU(A)) */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, lda * j + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + j);
            const double s = xhypot(Ajj_real, Ajj_imag);
            const double a_real = Ajj_real / s;
            const double a_imag = Ajj_imag / s;
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            for (k = j + 1; k < n2; k++) {
              const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
              const double Ajk_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + k);
              REAL_DOUBLE(B, ldb * i + k) -= Ajk_real * Bij_real - Ajk_imag * Bij_imag;
              IMAG_DOUBLE(B, ldb * i + k) -= Ajk_real * Bij_imag + Ajk_imag * Bij_real;
            }
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper && trans == CblasTrans) {

      /* form  B := alpha * B * inv(TriU(A))' */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, lda * j + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + j);
            const double s = xhypot(Ajj_real, Ajj_imag);
            const double a_real = Ajj_real / s;
            const double a_imag = Ajj_imag / s;
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            for (k = 0; k < j; k++) {
              const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
              const double Akj_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + j);
              REAL_DOUBLE(B, ldb * i + k) -= Akj_real * Bij_real - Akj_imag * Bij_imag;
              IMAG_DOUBLE(B, ldb * i + k) -= Akj_real * Bij_imag + Akj_imag * Bij_real;
            }
          }
        }
      }


    } else if (side == CblasRight && uplo == CblasLower && trans == CblasNoTrans) {

      /* form  B := alpha * B * inv(TriL(A)) */

      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = n2; j > 0 && j--;) {

          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, lda * j + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + j);
            const double s = xhypot(Ajj_real, Ajj_imag);
            const double a_real = Ajj_real / s;
            const double a_imag = Ajj_imag / s;
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            for (k = 0; k < j; k++) {
              const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
              const double Ajk_imag = conj * CONST_IMAG_DOUBLE(A, j * lda + k);
              REAL_DOUBLE(B, ldb * i + k) -= Ajk_real * Bij_real - Ajk_imag * Bij_imag;
              IMAG_DOUBLE(B, ldb * i + k) -= Ajk_real * Bij_imag + Ajk_imag * Bij_real;
            }
          }
        }
      }

    } else if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {

      /* form  B := alpha * B * inv(TriL(A))' */


      if (!(alpha_real == 1.0 && alpha_imag == 0.0)) {
        for (i = 0; i < n1; i++) {
          for (j = 0; j < n2; j++) {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = alpha_real * Bij_real - alpha_imag * Bij_imag;
            IMAG_DOUBLE(B, ldb * i + j) = alpha_real * Bij_imag + alpha_imag * Bij_real;
          }
        }
      }

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          if (nonunit) {
            const double Ajj_real = CONST_REAL_DOUBLE(A, lda * j + j);
            const double Ajj_imag = conj * CONST_IMAG_DOUBLE(A, lda * j + j);
            const double s = xhypot(Ajj_real, Ajj_imag);
            const double a_real = Ajj_real / s;
            const double a_imag = Ajj_imag / s;
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);
            REAL_DOUBLE(B, ldb * i + j) = (Bij_real * a_real + Bij_imag * a_imag) / s;
            IMAG_DOUBLE(B, ldb * i + j) = (Bij_imag * a_real - Bij_real * a_imag) / s;
          }

          {
            const double Bij_real = REAL_DOUBLE(B, ldb * i + j);
            const double Bij_imag = IMAG_DOUBLE(B, ldb * i + j);

            for (k = j + 1; k < n2; k++) {
              const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
              const double Akj_imag = conj * CONST_IMAG_DOUBLE(A, k * lda + j);
              REAL_DOUBLE(B, ldb * i + k) -= Akj_real * Bij_real - Akj_imag * Bij_imag;
              IMAG_DOUBLE(B, ldb * i + k) -= Akj_real * Bij_imag + Akj_imag * Bij_real;
            }
          }
        }
      }


    } else {
      fprintf(stderr, "unrecognized operation for cblas_ztrsm()\n");
    }
  }
}


void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int uplo, side;

  CHECK_ARGS13(HEMM,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    const float beta_real = CONST_REAL0_FLOAT(beta);
    const float beta_imag = CONST_IMAG0_FLOAT(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
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
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_FLOAT(C, ldc * i + j) = 0.0;
          IMAG_FLOAT(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Cij_real = REAL_FLOAT(C, ldc * i + j);
          const float Cij_imag = IMAG_FLOAT(C, ldc * i + j);
          REAL_FLOAT(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_FLOAT(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (side == CblasLeft && uplo == CblasUpper) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            /* const float Aii_imag = 0.0; */
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Aii_real;
            IMAG_FLOAT(C, i * ldc + j) += temp1_imag * Aii_real;
          }
          for (k = i + 1; k < n1; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = CONST_REAL_FLOAT(B, ldb * k + j);
            const float Bkj_imag = CONST_IMAG_FLOAT(B, ldb * k + j);
            REAL_FLOAT(C, k * ldc + j) += Aik_real * temp1_real - (-Aik_imag) * temp1_imag;
            IMAG_FLOAT(C, k * ldc + j) += Aik_real * temp1_imag + (-Aik_imag) * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          for (k = 0; k < i; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            const float Bkj_real = CONST_REAL_FLOAT(B, ldb * k + j);
            const float Bkj_imag = CONST_IMAG_FLOAT(B, ldb * k + j);
            REAL_FLOAT(C, k * ldc + j) += Aik_real * temp1_real - (-Aik_imag) * temp1_imag;
            IMAG_FLOAT(C, k * ldc + j) += Aik_real * temp1_imag + (-Aik_imag) * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          {
            const float Aii_real = CONST_REAL_FLOAT(A, i * lda + i);
            /* const float Aii_imag = 0.0; */
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Aii_real;
            IMAG_FLOAT(C, i * ldc + j) += temp1_imag * Aii_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            /* const float Ajj_imag = 0.0; */
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Ajj_real;
            IMAG_FLOAT(C, i * ldc + j) += temp1_imag * Ajj_real;
          }
          for (k = j + 1; k < n2; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, ldb * i + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, ldb * i + k);
            REAL_FLOAT(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_FLOAT(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * (-Ajk_imag);
            temp2_imag += Bik_real * (-Ajk_imag) + Bik_imag * Ajk_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const float Bij_real = CONST_REAL_FLOAT(B, ldb * i + j);
          const float Bij_imag = CONST_IMAG_FLOAT(B, ldb * i + j);
          const float temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const float temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          float temp2_real = 0.0;
          float temp2_imag = 0.0;
          for (k = 0; k < j; k++) {
            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            const float Bik_real = CONST_REAL_FLOAT(B, ldb * i + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, ldb * i + k);
            REAL_FLOAT(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_FLOAT(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * (-Ajk_imag);
            temp2_imag += Bik_real * (-Ajk_imag) + Bik_imag * Ajk_real;
          }
          {
            const float Ajj_real = CONST_REAL_FLOAT(A, j * lda + j);
            /* const float Ajj_imag = 0.0; */
            REAL_FLOAT(C, i * ldc + j) += temp1_real * Ajj_real;
            IMAG_FLOAT(C, i * ldc + j) += temp1_imag * Ajj_real;
          }
          REAL_FLOAT(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_FLOAT(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_chemm()\n");
    }
  }
}

void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const void *A, const int lda,
                 const float beta, void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(HERK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  if (beta == 1.0 && (alpha == 0.0 || K == 0))
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (Trans == CblasNoTrans) ? CblasConjTrans : CblasNoTrans;
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          REAL_FLOAT(C, ldc * i + j) = 0.0;
          IMAG_FLOAT(C, ldc * i + j) = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          REAL_FLOAT(C, ldc * i + j) = 0.0;
          IMAG_FLOAT(C, ldc * i + j) = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        REAL_FLOAT(C, ldc * i + i) *= beta;
        IMAG_FLOAT(C, ldc * i + i) = 0;
        for (j = i + 1; j < N; j++) {
          REAL_FLOAT(C, ldc * i + j) *= beta;
          IMAG_FLOAT(C, ldc * i + j) *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {
          REAL_FLOAT(C, ldc * i + j) *= beta;
          IMAG_FLOAT(C, ldc * i + j) *= beta;
        }
        REAL_FLOAT(C, ldc * i + i) *= beta;
        IMAG_FLOAT(C, ldc * i + i) = 0;
      }
    }
  } else {
    /* set imaginary part of Aii to zero */
    for (i = 0; i < N; i++) {
      IMAG_FLOAT(C, ldc * i + i) = 0.0;
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        float temp_real = 0.0;
        float temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
          const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
          const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
          const float Ajk_imag = -CONST_IMAG_FLOAT(A, j * lda + k);
          temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
          temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
        }
        REAL_FLOAT(C, i * ldc + j) += alpha * temp_real;
        IMAG_FLOAT(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasConjTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        float temp_real = 0.0;
        float temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          const float Aki_imag = -CONST_IMAG_FLOAT(A, k * lda + i);
          const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
          const float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
          temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
          temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
        }
        REAL_FLOAT(C, i * ldc + j) += alpha * temp_real;
        IMAG_FLOAT(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        float temp_real = 0.0;
        float temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
          const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
          const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
          const float Ajk_imag = -CONST_IMAG_FLOAT(A, j * lda + k);
          temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
          temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
        }
        REAL_FLOAT(C, i * ldc + j) += alpha * temp_real;
        IMAG_FLOAT(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasLower && trans == CblasConjTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        float temp_real = 0.0;
        float temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          const float Aki_imag = -CONST_IMAG_FLOAT(A, k * lda + i);
          const float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
          const float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
          temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
          temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
        }
        REAL_FLOAT(C, i * ldc + j) += alpha * temp_real;
        IMAG_FLOAT(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_cherk()\n");
  }
}


void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const float beta,
                  void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(HER2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (beta == 1.0 && ((alpha_real == 0.0 && alpha_imag == 0.0) || K == 0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      trans = Trans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasConjTrans : CblasNoTrans;
      alpha_imag *= -1;           /* conjugate alpha */
    }

    /* form  C := beta*C */

    if (beta == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_FLOAT(C, ldc * i + j) = 0.0;
            IMAG_FLOAT(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (beta != 1.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          REAL_FLOAT(C, ldc * i + i) *= beta;
          IMAG_FLOAT(C, ldc * i + i) = 0.0;
          for (j = i + 1; j < N; j++) {
            REAL_FLOAT(C, ldc * i + j) *= beta;
            IMAG_FLOAT(C, ldc * i + j) *= beta;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j < i; j++) {
            REAL_FLOAT(C, ldc * i + j) *= beta;
            IMAG_FLOAT(C, ldc * i + j) *= beta;
          }
          REAL_FLOAT(C, ldc * i + i) *= beta;
          IMAG_FLOAT(C, ldc * i + i) = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        IMAG_FLOAT(C, ldc * i + i) = 0.0;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {

        /* Cii += alpha Aik conj(Bik) + conj(alpha) Bik conj(Aik) */
        {
          float temp_real = 0.0;
          /* float temp_imag = 0.0; */
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const float temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const float temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);
            temp_real += temp1_real * Bik_real + temp1_imag * Bik_imag;
          }

          REAL_FLOAT(C, i * ldc + i) += 2 * temp_real;
          IMAG_FLOAT(C, i * ldc + i) = 0.0;
        }

        /* Cij += alpha Aik conj(Bjk) + conj(alpha) Bik conj(Ajk) */
        for (j = i + 1; j < N; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const float temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const float temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);

            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            /* temp2 = alpha * Ajk */
            const float temp2_real = alpha_real * Ajk_real - alpha_imag * Ajk_imag;
            const float temp2_imag = alpha_real * Ajk_imag + alpha_imag * Ajk_real;
            const float Bjk_real = CONST_REAL_FLOAT(B, j * ldb + k);
            const float Bjk_imag = CONST_IMAG_FLOAT(B, j * ldb + k);

            /* Cij += alpha * Aik * conj(Bjk) + conj(alpha) * Bik * conj(Ajk) */
            temp_real += ((temp1_real * Bjk_real + temp1_imag * Bjk_imag)
                          + (Bik_real * temp2_real + Bik_imag * temp2_imag));
            temp_imag += ((temp1_real * (-Bjk_imag) + temp1_imag * Bjk_real)
                          + (Bik_real * (-temp2_imag) + Bik_imag * temp2_real));
          }
          REAL_FLOAT(C, i * ldc + j) += temp_real;
          IMAG_FLOAT(C, i * ldc + j) += temp_imag;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasConjTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
          float Bki_real = CONST_REAL_FLOAT(B, k * ldb + i);
          float Bki_imag = CONST_IMAG_FLOAT(B, k * ldb + i);
          /* temp1 = alpha * conj(Aki) */
          float temp1_real = alpha_real * Aki_real - alpha_imag * (-Aki_imag);
          float temp1_imag = alpha_real * (-Aki_imag) + alpha_imag * Aki_real;
          /* temp2 = conj(alpha) * conj(Bki) */
          float temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          float temp2_imag = -(alpha_real * Bki_imag + alpha_imag * Bki_real);

          /* Cii += alpha * conj(Aki) * Bki + conj(alpha) * conj(Bki) * Aki */
          {
            REAL_FLOAT(C, i * lda + i) += 2 * (temp1_real * Bki_real - temp1_imag * Bki_imag);
            IMAG_FLOAT(C, i * lda + i) = 0.0;
          }

          for (j = i + 1; j < N; j++) {
            float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            float Bkj_real = CONST_REAL_FLOAT(B, k * ldb + j);
            float Bkj_imag = CONST_IMAG_FLOAT(B, k * ldb + j);
            /* Cij += alpha * conj(Aki) * Bkj + conj(alpha) * conj(Bki) * Akj */
            REAL_FLOAT(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_FLOAT(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {

        /* Cij += alpha Aik conj(Bjk) + conj(alpha) Bik conj(Ajk) */

        for (j = 0; j < i; j++) {
          float temp_real = 0.0;
          float temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const float temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const float temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);

            const float Ajk_real = CONST_REAL_FLOAT(A, j * lda + k);
            const float Ajk_imag = CONST_IMAG_FLOAT(A, j * lda + k);
            /* temp2 = alpha * Ajk */
            const float temp2_real = alpha_real * Ajk_real - alpha_imag * Ajk_imag;
            const float temp2_imag = alpha_real * Ajk_imag + alpha_imag * Ajk_real;
            const float Bjk_real = CONST_REAL_FLOAT(B, j * ldb + k);
            const float Bjk_imag = CONST_IMAG_FLOAT(B, j * ldb + k);

            /* Cij += alpha * Aik * conj(Bjk) + conj(alpha) * Bik * conj(Ajk) */
            temp_real += ((temp1_real * Bjk_real + temp1_imag * Bjk_imag)
                          + (Bik_real * temp2_real + Bik_imag * temp2_imag));
            temp_imag += ((temp1_real * (-Bjk_imag) + temp1_imag * Bjk_real)
                          + (Bik_real * (-temp2_imag) + Bik_imag * temp2_real));
          }
          REAL_FLOAT(C, i * ldc + j) += temp_real;
          IMAG_FLOAT(C, i * ldc + j) += temp_imag;
        }

        /* Cii += alpha Aik conj(Bik) + conj(alpha) Bik conj(Aik) */
        {
          float temp_real = 0.0;
          /* float temp_imag = 0.0; */
          for (k = 0; k < K; k++) {
            const float Aik_real = CONST_REAL_FLOAT(A, i * lda + k);
            const float Aik_imag = CONST_IMAG_FLOAT(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const float temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const float temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const float Bik_real = CONST_REAL_FLOAT(B, i * ldb + k);
            const float Bik_imag = CONST_IMAG_FLOAT(B, i * ldb + k);
            temp_real += temp1_real * Bik_real + temp1_imag * Bik_imag;
          }

          REAL_FLOAT(C, i * ldc + i) += 2 * temp_real;
          IMAG_FLOAT(C, i * ldc + i) = 0.0;
        }
      }

    } else if (uplo == CblasLower && trans == CblasConjTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          float Aki_real = CONST_REAL_FLOAT(A, k * lda + i);
          float Aki_imag = CONST_IMAG_FLOAT(A, k * lda + i);
          float Bki_real = CONST_REAL_FLOAT(B, k * ldb + i);
          float Bki_imag = CONST_IMAG_FLOAT(B, k * ldb + i);
          /* temp1 = alpha * conj(Aki) */
          float temp1_real = alpha_real * Aki_real - alpha_imag * (-Aki_imag);
          float temp1_imag = alpha_real * (-Aki_imag) + alpha_imag * Aki_real;
          /* temp2 = conj(alpha) * conj(Bki) */
          float temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          float temp2_imag = -(alpha_real * Bki_imag + alpha_imag * Bki_real);

          for (j = 0; j < i; j++) {
            float Akj_real = CONST_REAL_FLOAT(A, k * lda + j);
            float Akj_imag = CONST_IMAG_FLOAT(A, k * lda + j);
            float Bkj_real = CONST_REAL_FLOAT(B, k * ldb + j);
            float Bkj_imag = CONST_IMAG_FLOAT(B, k * ldb + j);
            /* Cij += alpha * conj(Aki) * Bkj + conj(alpha) * conj(Bki) * Akj */
            REAL_FLOAT(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_FLOAT(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }

          /* Cii += alpha * conj(Aki) * Bki + conj(alpha) * conj(Bki) * Aki */
          {
            REAL_FLOAT(C, i * lda + i) += 2 * (temp1_real * Bki_real - temp1_imag * Bki_imag);
            IMAG_FLOAT(C, i * lda + i) = 0.0;
          }
        }
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_cher2k()\n");
    }
  }
}

void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta,
                 void *C, const int ldc)
{
  int i, j, k;
  int n1, n2;
  int uplo, side;

  CHECK_ARGS13(HEMM,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    const double beta_real = CONST_REAL0_DOUBLE(beta);
    const double beta_imag = CONST_IMAG0_DOUBLE(beta);

    if ((alpha_real == 0.0 && alpha_imag == 0.0)
        && (beta_real == 1.0 && beta_imag == 0.0))
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
    if (beta_real == 0.0 && beta_imag == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          REAL_DOUBLE(C, ldc * i + j) = 0.0;
          IMAG_DOUBLE(C, ldc * i + j) = 0.0;
        }
      }
    } else if (!(beta_real == 1.0 && beta_imag == 0.0)) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Cij_real = REAL_DOUBLE(C, ldc * i + j);
          const double Cij_imag = IMAG_DOUBLE(C, ldc * i + j);
          REAL_DOUBLE(C, ldc * i + j) = beta_real * Cij_real - beta_imag * Cij_imag;
          IMAG_DOUBLE(C, ldc * i + j) = beta_real * Cij_imag + beta_imag * Cij_real;
        }
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (side == CblasLeft && uplo == CblasUpper) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            /* const double Aii_imag = 0.0; */
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Aii_real;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_imag * Aii_real;
          }
          for (k = i + 1; k < n1; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = CONST_REAL_DOUBLE(B, ldb * k + j);
            const double Bkj_imag = CONST_IMAG_DOUBLE(B, ldb * k + j);
            REAL_DOUBLE(C, k * ldc + j) += Aik_real * temp1_real - (-Aik_imag) * temp1_imag;
            IMAG_DOUBLE(C, k * ldc + j) += Aik_real * temp1_imag + (-Aik_imag) * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasLeft && uplo == CblasLower) {

      /* form  C := alpha*A*B + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          for (k = 0; k < i; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            const double Bkj_real = CONST_REAL_DOUBLE(B, ldb * k + j);
            const double Bkj_imag = CONST_IMAG_DOUBLE(B, ldb * k + j);
            REAL_DOUBLE(C, k * ldc + j) += Aik_real * temp1_real - (-Aik_imag) * temp1_imag;
            IMAG_DOUBLE(C, k * ldc + j) += Aik_real * temp1_imag + (-Aik_imag) * temp1_real;
            temp2_real += Aik_real * Bkj_real - Aik_imag * Bkj_imag;
            temp2_imag += Aik_real * Bkj_imag + Aik_imag * Bkj_real;
          }
          {
            const double Aii_real = CONST_REAL_DOUBLE(A, i * lda + i);
            /* const double Aii_imag = 0.0; */
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Aii_real;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_imag * Aii_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasUpper) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            /* const double Ajj_imag = 0.0; */
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_real;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_imag * Ajj_real;
          }
          for (k = j + 1; k < n2; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, ldb * i + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, ldb * i + k);
            REAL_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * (-Ajk_imag);
            temp2_imag += Bik_real * (-Ajk_imag) + Bik_imag * Ajk_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else if (side == CblasRight && uplo == CblasLower) {

      /* form  C := alpha*B*A + C */

      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          const double Bij_real = CONST_REAL_DOUBLE(B, ldb * i + j);
          const double Bij_imag = CONST_IMAG_DOUBLE(B, ldb * i + j);
          const double temp1_real = alpha_real * Bij_real - alpha_imag * Bij_imag;
          const double temp1_imag = alpha_real * Bij_imag + alpha_imag * Bij_real;
          double temp2_real = 0.0;
          double temp2_imag = 0.0;
          for (k = 0; k < j; k++) {
            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            const double Bik_real = CONST_REAL_DOUBLE(B, ldb * i + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, ldb * i + k);
            REAL_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_real - temp1_imag * Ajk_imag;
            IMAG_DOUBLE(C, i * ldc + k) += temp1_real * Ajk_imag + temp1_imag * Ajk_real;
            temp2_real += Bik_real * Ajk_real - Bik_imag * (-Ajk_imag);
            temp2_imag += Bik_real * (-Ajk_imag) + Bik_imag * Ajk_real;
          }
          {
            const double Ajj_real = CONST_REAL_DOUBLE(A, j * lda + j);
            /* const double Ajj_imag = 0.0; */
            REAL_DOUBLE(C, i * ldc + j) += temp1_real * Ajj_real;
            IMAG_DOUBLE(C, i * ldc + j) += temp1_imag * Ajj_real;
          }
          REAL_DOUBLE(C, i * ldc + j) += alpha_real * temp2_real - alpha_imag * temp2_imag;
          IMAG_DOUBLE(C, i * ldc + j) += alpha_real * temp2_imag + alpha_imag * temp2_real;
        }
      }

    } else {
      fprintf(stderr, "unrecognized operation for cblas_zhemm()\n");
    }
  }
}

void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const void *A, const int lda,
                 const double beta, void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS11(HERK,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);

  if (beta == 1.0 && (alpha == 0.0 || K == 0))
    return;

  if (Order == CblasRowMajor) {
    uplo = Uplo;
    trans = Trans;
  } else {
    uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
    trans = (Trans == CblasNoTrans) ? CblasConjTrans : CblasNoTrans;
  }

  /* form  y := beta*y */
  if (beta == 0.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
          REAL_DOUBLE(C, ldc * i + j) = 0.0;
          IMAG_DOUBLE(C, ldc * i + j) = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
          REAL_DOUBLE(C, ldc * i + j) = 0.0;
          IMAG_DOUBLE(C, ldc * i + j) = 0.0;
        }
      }
    }
  } else if (beta != 1.0) {
    if (uplo == CblasUpper) {
      for (i = 0; i < N; i++) {
        REAL_DOUBLE(C, ldc * i + i) *= beta;
        IMAG_DOUBLE(C, ldc * i + i) = 0;
        for (j = i + 1; j < N; j++) {
          REAL_DOUBLE(C, ldc * i + j) *= beta;
          IMAG_DOUBLE(C, ldc * i + j) *= beta;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {
          REAL_DOUBLE(C, ldc * i + j) *= beta;
          IMAG_DOUBLE(C, ldc * i + j) *= beta;
        }
        REAL_DOUBLE(C, ldc * i + i) *= beta;
        IMAG_DOUBLE(C, ldc * i + i) = 0;
      }
    }
  } else {
    /* set imaginary part of Aii to zero */
    for (i = 0; i < N; i++) {
      IMAG_DOUBLE(C, ldc * i + i) = 0.0;
    }
  }

  if (alpha == 0.0)
    return;

  if (uplo == CblasUpper && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        double temp_real = 0.0;
        double temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
          const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
          const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
          const double Ajk_imag = -CONST_IMAG_DOUBLE(A, j * lda + k);
          temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
          temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
        }
        REAL_DOUBLE(C, i * ldc + j) += alpha * temp_real;
        IMAG_DOUBLE(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasUpper && trans == CblasConjTrans) {

    for (i = 0; i < N; i++) {
      for (j = i; j < N; j++) {
        double temp_real = 0.0;
        double temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          const double Aki_imag = -CONST_IMAG_DOUBLE(A, k * lda + i);
          const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
          const double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
          temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
          temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
        }
        REAL_DOUBLE(C, i * ldc + j) += alpha * temp_real;
        IMAG_DOUBLE(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasLower && trans == CblasNoTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        double temp_real = 0.0;
        double temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
          const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
          const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
          const double Ajk_imag = -CONST_IMAG_DOUBLE(A, j * lda + k);
          temp_real += Aik_real * Ajk_real - Aik_imag * Ajk_imag;
          temp_imag += Aik_real * Ajk_imag + Aik_imag * Ajk_real;
        }
        REAL_DOUBLE(C, i * ldc + j) += alpha * temp_real;
        IMAG_DOUBLE(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else if (uplo == CblasLower && trans == CblasConjTrans) {

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++) {
        double temp_real = 0.0;
        double temp_imag = 0.0;
        for (k = 0; k < K; k++) {
          const double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          const double Aki_imag = -CONST_IMAG_DOUBLE(A, k * lda + i);
          const double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
          const double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
          temp_real += Aki_real * Akj_real - Aki_imag * Akj_imag;
          temp_imag += Aki_real * Akj_imag + Aki_imag * Akj_real;
        }
        REAL_DOUBLE(C, i * ldc + j) += alpha * temp_real;
        IMAG_DOUBLE(C, i * ldc + j) += alpha * temp_imag;
      }
    }

  } else {
    fprintf(stderr, "unrecognized operation for cblas_zherk()\n");
  }
}


void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb, const double beta,
                  void *C, const int ldc)
{
  int i, j, k;
  int uplo, trans;

  CHECK_ARGS13(HER2K,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

  {
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (beta == 1.0 && ((alpha_real == 0.0 && alpha_imag == 0.0) || K == 0))
      return;

    if (Order == CblasRowMajor) {
      uplo = Uplo;
      trans = Trans;
    } else {
      uplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
      trans = (Trans == CblasNoTrans) ? CblasConjTrans : CblasNoTrans;
      alpha_imag *= -1;           /* conjugate alpha */
    }

    /* form  C := beta*C */

    if (beta == 0.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          for (j = i; j < N; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j <= i; j++) {
            REAL_DOUBLE(C, ldc * i + j) = 0.0;
            IMAG_DOUBLE(C, ldc * i + j) = 0.0;
          }
        }
      }
    } else if (beta != 1.0) {
      if (uplo == CblasUpper) {
        for (i = 0; i < N; i++) {
          REAL_DOUBLE(C, ldc * i + i) *= beta;
          IMAG_DOUBLE(C, ldc * i + i) = 0.0;
          for (j = i + 1; j < N; j++) {
            REAL_DOUBLE(C, ldc * i + j) *= beta;
            IMAG_DOUBLE(C, ldc * i + j) *= beta;
          }
        }
      } else {
        for (i = 0; i < N; i++) {
          for (j = 0; j < i; j++) {
            REAL_DOUBLE(C, ldc * i + j) *= beta;
            IMAG_DOUBLE(C, ldc * i + j) *= beta;
          }
          REAL_DOUBLE(C, ldc * i + i) *= beta;
          IMAG_DOUBLE(C, ldc * i + i) = 0.0;
        }
      }
    } else {
      for (i = 0; i < N; i++) {
        IMAG_DOUBLE(C, ldc * i + i) = 0.0;
      }
    }

    if (alpha_real == 0.0 && alpha_imag == 0.0)
      return;

    if (uplo == CblasUpper && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {

        /* Cii += alpha Aik conj(Bik) + conj(alpha) Bik conj(Aik) */
        {
          double temp_real = 0.0;
          /* double temp_imag = 0.0; */
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const double temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const double temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);
            temp_real += temp1_real * Bik_real + temp1_imag * Bik_imag;
          }

          REAL_DOUBLE(C, i * ldc + i) += 2 * temp_real;
          IMAG_DOUBLE(C, i * ldc + i) = 0.0;
        }

        /* Cij += alpha Aik conj(Bjk) + conj(alpha) Bik conj(Ajk) */
        for (j = i + 1; j < N; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const double temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const double temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);

            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            /* temp2 = alpha * Ajk */
            const double temp2_real = alpha_real * Ajk_real - alpha_imag * Ajk_imag;
            const double temp2_imag = alpha_real * Ajk_imag + alpha_imag * Ajk_real;
            const double Bjk_real = CONST_REAL_DOUBLE(B, j * ldb + k);
            const double Bjk_imag = CONST_IMAG_DOUBLE(B, j * ldb + k);

            /* Cij += alpha * Aik * conj(Bjk) + conj(alpha) * Bik * conj(Ajk) */
            temp_real += ((temp1_real * Bjk_real + temp1_imag * Bjk_imag)
                          + (Bik_real * temp2_real + Bik_imag * temp2_imag));
            temp_imag += ((temp1_real * (-Bjk_imag) + temp1_imag * Bjk_real)
                          + (Bik_real * (-temp2_imag) + Bik_imag * temp2_real));
          }
          REAL_DOUBLE(C, i * ldc + j) += temp_real;
          IMAG_DOUBLE(C, i * ldc + j) += temp_imag;
        }
      }

    } else if (uplo == CblasUpper && trans == CblasConjTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
          double Bki_real = CONST_REAL_DOUBLE(B, k * ldb + i);
          double Bki_imag = CONST_IMAG_DOUBLE(B, k * ldb + i);
          /* temp1 = alpha * conj(Aki) */
          double temp1_real = alpha_real * Aki_real - alpha_imag * (-Aki_imag);
          double temp1_imag = alpha_real * (-Aki_imag) + alpha_imag * Aki_real;
          /* temp2 = conj(alpha) * conj(Bki) */
          double temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          double temp2_imag = -(alpha_real * Bki_imag + alpha_imag * Bki_real);

          /* Cii += alpha * conj(Aki) * Bki + conj(alpha) * conj(Bki) * Aki */
          {
            REAL_DOUBLE(C, i * lda + i) += 2 * (temp1_real * Bki_real - temp1_imag * Bki_imag);
            IMAG_DOUBLE(C, i * lda + i) = 0.0;
          }

          for (j = i + 1; j < N; j++) {
            double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            double Bkj_real = CONST_REAL_DOUBLE(B, k * ldb + j);
            double Bkj_imag = CONST_IMAG_DOUBLE(B, k * ldb + j);
            /* Cij += alpha * conj(Aki) * Bkj + conj(alpha) * conj(Bki) * Akj */
            REAL_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }
        }
      }

    } else if (uplo == CblasLower && trans == CblasNoTrans) {

      for (i = 0; i < N; i++) {

        /* Cij += alpha Aik conj(Bjk) + conj(alpha) Bik conj(Ajk) */

        for (j = 0; j < i; j++) {
          double temp_real = 0.0;
          double temp_imag = 0.0;
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const double temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const double temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);

            const double Ajk_real = CONST_REAL_DOUBLE(A, j * lda + k);
            const double Ajk_imag = CONST_IMAG_DOUBLE(A, j * lda + k);
            /* temp2 = alpha * Ajk */
            const double temp2_real = alpha_real * Ajk_real - alpha_imag * Ajk_imag;
            const double temp2_imag = alpha_real * Ajk_imag + alpha_imag * Ajk_real;
            const double Bjk_real = CONST_REAL_DOUBLE(B, j * ldb + k);
            const double Bjk_imag = CONST_IMAG_DOUBLE(B, j * ldb + k);

            /* Cij += alpha * Aik * conj(Bjk) + conj(alpha) * Bik * conj(Ajk) */
            temp_real += ((temp1_real * Bjk_real + temp1_imag * Bjk_imag)
                          + (Bik_real * temp2_real + Bik_imag * temp2_imag));
            temp_imag += ((temp1_real * (-Bjk_imag) + temp1_imag * Bjk_real)
                          + (Bik_real * (-temp2_imag) + Bik_imag * temp2_real));
          }
          REAL_DOUBLE(C, i * ldc + j) += temp_real;
          IMAG_DOUBLE(C, i * ldc + j) += temp_imag;
        }

        /* Cii += alpha Aik conj(Bik) + conj(alpha) Bik conj(Aik) */
        {
          double temp_real = 0.0;
          /* double temp_imag = 0.0; */
          for (k = 0; k < K; k++) {
            const double Aik_real = CONST_REAL_DOUBLE(A, i * lda + k);
            const double Aik_imag = CONST_IMAG_DOUBLE(A, i * lda + k);
            /* temp1 = alpha * Aik */
            const double temp1_real = alpha_real * Aik_real - alpha_imag * Aik_imag;
            const double temp1_imag = alpha_real * Aik_imag + alpha_imag * Aik_real;
            const double Bik_real = CONST_REAL_DOUBLE(B, i * ldb + k);
            const double Bik_imag = CONST_IMAG_DOUBLE(B, i * ldb + k);
            temp_real += temp1_real * Bik_real + temp1_imag * Bik_imag;
          }

          REAL_DOUBLE(C, i * ldc + i) += 2 * temp_real;
          IMAG_DOUBLE(C, i * ldc + i) = 0.0;
        }
      }

    } else if (uplo == CblasLower && trans == CblasConjTrans) {

      for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++) {
          double Aki_real = CONST_REAL_DOUBLE(A, k * lda + i);
          double Aki_imag = CONST_IMAG_DOUBLE(A, k * lda + i);
          double Bki_real = CONST_REAL_DOUBLE(B, k * ldb + i);
          double Bki_imag = CONST_IMAG_DOUBLE(B, k * ldb + i);
          /* temp1 = alpha * conj(Aki) */
          double temp1_real = alpha_real * Aki_real - alpha_imag * (-Aki_imag);
          double temp1_imag = alpha_real * (-Aki_imag) + alpha_imag * Aki_real;
          /* temp2 = conj(alpha) * conj(Bki) */
          double temp2_real = alpha_real * Bki_real - alpha_imag * Bki_imag;
          double temp2_imag = -(alpha_real * Bki_imag + alpha_imag * Bki_real);

          for (j = 0; j < i; j++) {
            double Akj_real = CONST_REAL_DOUBLE(A, k * lda + j);
            double Akj_imag = CONST_IMAG_DOUBLE(A, k * lda + j);
            double Bkj_real = CONST_REAL_DOUBLE(B, k * ldb + j);
            double Bkj_imag = CONST_IMAG_DOUBLE(B, k * ldb + j);
            /* Cij += alpha * conj(Aki) * Bkj + conj(alpha) * conj(Bki) * Akj */
            REAL_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_real - temp1_imag * Bkj_imag)
              + (temp2_real * Akj_real - temp2_imag * Akj_imag);
            IMAG_DOUBLE(C, i * lda + j) += (temp1_real * Bkj_imag + temp1_imag * Bkj_real)
              + (temp2_real * Akj_imag + temp2_imag * Akj_real);
          }

          /* Cii += alpha * conj(Aki) * Bki + conj(alpha) * conj(Bki) * Aki */
          {
            REAL_DOUBLE(C, i * lda + i) += 2 * (temp1_real * Bki_real - temp1_imag * Bki_imag);
            IMAG_DOUBLE(C, i * lda + i) = 0.0;
          }
        }
      }
    } else {
      fprintf(stderr, "unrecognized operation for cblas_zher2k()\n");
    }
  }
}

