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

#ifndef _NA_CBLAS_ERROR_H_
#define _NA_CBLAS_ERROR_H_          1

#include "na_cblas.h"

// Error check marcos
#define CHECK_ARGS_X(FUNCTION,VAR,ARGS) do { int VAR = 0 ;      \
    CBLAS_ERROR_##FUNCTION ARGS ; \
    if (VAR) cblas_xerbla(pos,__FILE__,""); } while (0)

#define CHECK_ARGS7(FUNCTION,A1,A2,A3,A4,A5,A6,A7) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7))
#define CHECK_ARGS8(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8))
#define CHECK_ARGS9(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8,A9) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8,A9))
#define CHECK_ARGS10(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10))
#define CHECK_ARGS11(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11))
#define CHECK_ARGS12(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12))
#define CHECK_ARGS14(FUNCTION,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14) \
  CHECK_ARGS_X(FUNCTION,pos,(pos,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14))

/* check if CBLAS_ORDER is correct */
#define CHECK_ORDER(pos,posIfError,order) \
if(((order)!=CblasRowMajor)&&((order)!=CblasColMajor)) \
     pos = posIfError;
/* check if CBLAS_TRANSPOSE is correct */
#define CHECK_TRANSPOSE(pos,posIfError,Trans) \
if(((Trans)!=CblasNoTrans)&&((Trans)!=CblasTrans)&&((Trans)!=CblasConjTrans)) \
    pos = posIfError;
/* check if CBLAS_UPLO is correct */
#define CHECK_UPLO(pos,posIfError,Uplo) \
if(((Uplo)!=CblasUpper)&&((Uplo)!=CblasLower)) \
    pos = posIfError;
/* check if CBLAS_DIAG is correct */
#define CHECK_DIAG(pos,posIfError,Diag) \
if(((Diag)!=CblasNonUnit)&&((Diag)!=CblasUnit)) \
    pos = posIfError;
/* check if CBLAS_SIDE is correct */
#define CHECK_SIDE(pos,posIfError,Side) \
if(((Side)!=CblasLeft)&&((Side)!=CblasRight)) \
    pos = posIfError;
/* check if a dimension argument is correct */
#define CHECK_DIM(pos,posIfError,dim) \
if((dim)<0) \
    pos = posIfError;
/* check if a stride argument is correct */
#define CHECK_STRIDE(pos,posIfError,stride) \
if((stride)==0) \
    pos = posIfError;

/* cblas_xgemv() */
#define CBLAS_ERROR_GEMV(pos,order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_TRANSPOSE(pos,2,TransA); \
CHECK_DIM(pos,3,M); \
CHECK_DIM(pos,4,N); \
if((order)==CblasRowMajor) { \
    if((lda)<NA_MAX(1,N)) { \
        pos = 7; \
    } \
} else if((order)==CblasColMajor) { \
    if((lda)<NA_MAX(1,M)) { \
        pos = 7; \
    } \
};                       \
CHECK_STRIDE(pos,9,incX); \
CHECK_STRIDE(pos,12,incY);

/* cblas_xgbmv() */
#define CBLAS_ERROR_GBMV(pos,order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_TRANSPOSE(pos,2,TransA); \
CHECK_DIM(pos,3,M); \
CHECK_DIM(pos,4,N); \
CHECK_DIM(pos,5,KL); \
CHECK_DIM(pos,6,KU); \
if((lda)<NA_MAX(1,(KL+KU+1))) { \
    pos = 9; \
};                        \
CHECK_STRIDE(pos,11,incX); \
CHECK_STRIDE(pos,14,incY);

/* cblas_xtrmv() */
#define CBLAS_ERROR_TRMV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
if((lda)<NA_MAX(1,N)) \
    pos = 7; \
CHECK_STRIDE(pos,9,incX);

/* cblas_xtbmv() */
#define CBLAS_ERROR_TBMV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
CHECK_DIM(pos,6,K); \
if((lda)<NA_MAX(1,(K+1))) { \
    pos = 8; \
}; \
CHECK_STRIDE(pos,10,incX);

/* cblas_xtpmv() */
#define CBLAS_ERROR_TPMV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
CHECK_STRIDE(pos,8,incX);

/* cblas_xtrsv() */
#define CBLAS_ERROR_TRSV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX) \
CBLAS_ERROR_TRMV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX)

/* cblas_xtbsv() */
#define CBLAS_ERROR_TBSV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX) \
CBLAS_ERROR_TBMV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX)

/* cblas_xtpsv() */
#define CBLAS_ERROR_TPSV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX) \
CBLAS_ERROR_TPMV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX)

/* cblas_xsymv() */
#define CBLAS_ERROR_SD_SYMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
if((lda)<NA_MAX(1,N)) { \
    pos = 6; \
};                       \
CHECK_STRIDE(pos,8,incX); \
CHECK_STRIDE(pos,11,incY);

/* cblas_xsbmv() */
#define CBLAS_ERROR_SD_SBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_DIM(pos,4,K); \
if((lda)<NA_MAX(1,K+1)) { \
    pos = 7; \
};                       \
CHECK_STRIDE(pos,9,incX); \
CHECK_STRIDE(pos,12,incY);

/* cblas_xspmv() */
#define CBLAS_ERROR_SD_SPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,7,incX); \
CHECK_STRIDE(pos,10,incY);

/* cblas_xger() */
#define CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_DIM(pos,2,M); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY); \
if((order)==CblasRowMajor) { \
    if((lda)<NA_MAX(1,N)) { \
        pos = 10; \
    } \
} else if((order)==CblasColMajor) { \
    if((lda)<NA_MAX(1,M)) { \
        pos = 10; \
    } \
};

/* cblas_xsyr() */
#define CBLAS_ERROR_SD_SYR(pos,order,Uplo,N,alpha,X,incX,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
if((lda)<NA_MAX(1,N)) { \
    pos = 8; \
};

/* cblas_xspr() */
#define CBLAS_ERROR_SD_SPR(pos,order,Uplo,N,alpha,X,incX,Ap) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX);

/* cblas_xsyr2() */
#define CBLAS_ERROR_SD_SYR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY); \
if((lda)<NA_MAX(1,N)) { \
    pos = 10; \
};

/* cblas_xspr2() */
#define CBLAS_ERROR_SD_SPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY);

/*
 * Routines with C and Z prefixes only
 */
/* cblas_xhemv() */
#define CBLAS_ERROR_CZ_HEMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SYMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY)

/* cblas_xhbmv() */
#define CBLAS_ERROR_CZ_HBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY)

/* cblas_xhpmv() */
#define CBLAS_ERROR_CZ_HPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY)

/* cblas_xgeru() */
#define CBLAS_ERROR_CZ_GERU(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xgerc() */
#define CBLAS_ERROR_CZ_GERC(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xher() */
#define CBLAS_ERROR_CZ_HER(pos,order,Uplo,N,alpha,X,incX,A,lda) \
CBLAS_ERROR_SD_SYR(pos,order,Uplo,N,alpha,X,incX,A,lda)

/* cblas_xhpr() */
#define CBLAS_ERROR_CZ_HPR(pos,order,Uplo,N,alpha,X,incX,A) \
CBLAS_ERROR_SD_SPR(pos,order,Uplo,N,alpha,X,incX,A)

/* cblas_xher2() */
#define CBLAS_ERROR_CZ_HER2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_SYR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xhpr2() */
#define CBLAS_ERROR_CZ_HPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,Ap) \
CBLAS_ERROR_SD_SPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,Ap)

#endif
