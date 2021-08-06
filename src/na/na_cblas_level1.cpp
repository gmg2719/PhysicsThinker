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
#include <cstdlib>
#include <cmath>
#include "na/na_cblas.h"

/*
 * ===========================================================================
 * Prototypes for some BLAS common functions
 * ===========================================================================
 */
void cblas_xerbla(int p, const char *rout, const char *form, ...)
{
    va_list ap;
    va_start (ap, form);

    if (p)
    {
        fprintf (stderr, "Parameter %d to routine %s was incorrect\n", p, rout);
    }

    vfprintf (stderr, form, ap);
    va_end (ap);
    abort ();
}

float cblas_sdsdot(const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY)
{
    double r = alpha;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        r += X[ix] * Y[iy];
        ix += incX;
        iy += incY;
    }

    return float(r);
}

double cblas_dsdot(const int N, const float *X, const int incX, const float *Y,
                   const int incY)
{
    double r = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        r += X[ix] * Y[iy];
        ix += incX;
        iy += incY;
    }

    return r;
}

float  cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY)
{
    float r = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        r += X[ix] * Y[iy];
        ix += incX;
        iy += incY;
    }

    return r;
}

double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY)
{
    double r = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        r += X[ix] * Y[iy];
        ix += incX;
        iy += incY;
    }

    return r;
}

void   cblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
    float r_real = 0.0;
    float r_imag = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (int i = 0; i < N; i++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float y_real = CONST_REAL_FLOAT(Y, iy);
        const float y_imag = CONST_IMAG_FLOAT(Y, iy);
        // CONJ = 1.0
        r_real += x_real * y_real - x_imag * y_imag;
        r_imag += x_real * y_imag + x_imag * y_real;
        ix += incX;
        iy += incY;
    }
    REAL0_FLOAT(dotu) = r_real;
    IMAG0_FLOAT(dotu) = r_imag;
}

void   cblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
    float r_real = 0.0;
    float r_imag = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (int i = 0; i < N; i++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        const float y_real = CONST_REAL_FLOAT(Y, iy);
        const float y_imag = CONST_IMAG_FLOAT(Y, iy);
        // CONJ = -1.0
        r_real += x_real * y_real + x_imag * y_imag;
        r_imag += x_real * y_imag - x_imag * y_real;
        ix += incX;
        iy += incY;
    }
    REAL0_FLOAT(dotc) = r_real;
    IMAG0_FLOAT(dotc) = r_imag;
}

void   cblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
    double r_real = 0.0;
    double r_imag = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double y_real = CONST_REAL_DOUBLE(Y, iy);
        const double y_imag = CONST_IMAG_DOUBLE(Y, iy);
        // CONJ = 1.0
        r_real += x_real * y_real - x_imag * y_imag;
        r_imag += x_real * y_imag + x_imag * y_real;
        ix += incX;
        iy += incY;
    }
    REAL0_DOUBLE(dotu) = r_real;
    IMAG0_DOUBLE(dotu) = r_imag;
}

void   cblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
    double r_real = 0.0;
    double r_imag = 0.0;
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        const double y_real = CONST_REAL_DOUBLE(Y, iy);
        const double y_imag = CONST_IMAG_DOUBLE(Y, iy);
        r_real += x_real * y_real + x_imag * y_imag;
        r_imag += x_real * y_imag - x_imag * y_real;
        ix += incX;
        iy += incY;
    }
    REAL0_DOUBLE(dotc) = r_real;
    IMAG0_DOUBLE(dotc) = r_imag;
}

float  cblas_snrm2(const int N, const float *X, const int incX)
{
    float scale = 0.0;
    float ssq = 1.0;
    int ix = 0;

    if (N <= 0 || incX <= 0) {
        return 0;
    } else if (N == 1) {
        return fabs(X[0]);
    }

    for (int i = 0; i < N; i++) {
        const float x = X[ix];

        if (x != 0.0) {
            const float ax = fabs(x);

            if (scale < ax) {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }
        ix += incX;
    }

    return scale * sqrt(ssq);
}

float  cblas_sasum(const int N, const float *X, const int incX)
{
    float r = 0.0;
    int ix = 0;
    
    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        r += fabs(X[ix]);
        ix += incX;
    }

    return r;
}

double cblas_dnrm2(const int N, const double *X, const int incX)
{
    double scale = 0.0;
    double ssq = 1.0;
    int ix = 0;

    if (N <= 0 || incX <= 0) {
        return 0;
    } else if (N == 1) {
        return fabs(X[0]);
    }

    for (int i = 0; i < N; i++) {
        const double x = X[ix];

        if (x != 0.0) {
            const double ax = fabs(x);

            if (scale < ax) {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }
        ix += incX;
    }

    return scale * sqrt(ssq);
}

double cblas_dasum(const int N, const double *X, const int incX)
{
    double r = 0.0;
    int ix = 0;
    
    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        r += fabs(X[ix]);
        ix += incX;
    }

    return r;
}

float  cblas_scnrm2(const int N, const void *X, const int incX)
{
    float scale = 0.0;
    float ssq = 1.0;
    int ix = 0;

    if (N == 0 || incX < 1) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        const float x = CONST_REAL_FLOAT(X, ix);
        const float y = CONST_IMAG_FLOAT(X, ix);

        if (x != 0.0) {
            const float ax = fabs(x);

            if (scale < ax) {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }

        if (y != 0.0) {
            const float ay = fabs(y);

            if (scale < ay) {
                ssq = 1.0 + ssq * (scale / ay) * (scale / ay);
                scale = ay;
            } else {
                ssq += (ay / scale) * (ay / scale);
            }
        }

        ix += incX;
    }

    return scale * sqrt(ssq);
}

float  cblas_scasum(const int N, const void *X, const int incX)
{
    float r = 0.0;
    int ix = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        r += fabs(CONST_REAL_FLOAT(X, ix)) + fabs(CONST_IMAG_FLOAT(X, ix));
        ix += incX;
    }

    return r;
}

double cblas_dznrm2(const int N, const void *X, const int incX)
{
    double scale = 0.0;
    double ssq = 1.0;
    int ix = 0;

    if (N == 0 || incX < 1) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        const double x = CONST_REAL_DOUBLE(X, ix);
        const double y = CONST_IMAG_DOUBLE(X, ix);

        if (x != 0.0) {
            const double ax = fabs(x);

            if (scale < ax) {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }

        if (y != 0.0) {
            const double ay = fabs(y);

            if (scale < ay) {
                ssq = 1.0 + ssq * (scale / ay) * (scale / ay);
                scale = ay;
            } else {
                ssq += (ay / scale) * (ay / scale);
            }
        }

        ix += incX;
    }

    return scale * sqrt(ssq);
}

double cblas_dzasum(const int N, const void *X, const int incX)
{
    double r = 0.0;
    int ix = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        r += fabs(CONST_REAL_DOUBLE(X, ix)) + fabs(CONST_IMAG_DOUBLE(X, ix));
        ix += incX;
    }

    return r;
}

size_t cblas_isamax(const int N, const float  *X, const int incX)
{
    float max_value = 0.0;
    int ix = 0;
    size_t result = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        if (fabs(X[ix]) > max_value) {
            max_value = fabs(X[ix]);
            result = i;
        }
        ix += incX;
    }

    return result;
}

size_t cblas_idamax(const int N, const double *X, const int incX)
{
    double max_value = 0.0;
    int ix = 0;
    size_t result = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        if (fabs(X[ix]) > max_value) {
            max_value = fabs(X[ix]);
            result = i;
        }
        ix += incX;
    }

    return result;
}

size_t cblas_icamax(const int N, const void   *X, const int incX)
{
    float max_value = 0.0;
    int ix = 0;
    size_t result = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        const float a = fabs(CONST_REAL_FLOAT(X, ix)) + fabs(CONST_IMAG_FLOAT(X, ix));

        if (a > max_value) {
            max_value = a;
            result = i;
        }
        ix += incX;
    }

    return result;
}

size_t cblas_izamax(const int N, const void   *X, const int incX)
{
    double max_value = 0.0;
    int ix = 0;
    size_t result = 0;

    if (incX <= 0) {
        return 0;
    }

    for (int i = 0; i < N; i++) {
        const double a = fabs(CONST_REAL_DOUBLE(X, ix)) + fabs(CONST_IMAG_DOUBLE(X, ix));

        if (a > max_value) {
            max_value = a;
            result = i;
        }
        ix += incX;
    }

    return result;
}

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
void cblas_sswap(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const float tmp = X[ix];
        X[ix] = Y[iy];
        Y[iy] = tmp;
        ix += incX;
        iy += incY;
    }
}

void cblas_scopy(const int N, const float *X, const int incX, 
                 float *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        Y[iy] = X[ix];
        ix += incX;
        iy += incY;
    }
}

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
    if (alpha == 0.0) {
        return;
    }

    if (incX == 1 && incY == 1) {
        const int m = N % 4;

        for (int i = 0; i < m; i++) {
            Y[i] += alpha * X[i];
        }

        for (int i = m; i + 3 < N; i += 4) {
            Y[i] += alpha * X[i];
            Y[i + 1] += alpha * X[i + 1];
            Y[i + 2] += alpha * X[i + 2];
            Y[i + 3] += alpha * X[i + 3];
        }
    }
    else
    {
        int ix = OFFSET(N, incX);
        int iy = OFFSET(N, incY);

        for (int i = 0; i < N; i++) {
            Y[iy] += alpha * X[ix];
            ix += incX;
            iy += incY;
        }
    }
}

void cblas_dswap(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const double tmp = X[ix];
        X[ix] = Y[iy];
        Y[iy] = tmp;
        ix += incX;
        iy += incY;
    }
}


void cblas_dcopy(const int N, const double *X, const int incX, 
                 double *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        Y[iy] = X[ix];
        ix += incX;
        iy += incY;
    }
}


void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY)
{
    if (alpha == 0.0) {
        return;
    }

    if (incX == 1 && incY == 1) {
        const int m = N % 4;

        for (int i = 0; i < m; i++) {
            Y[i] += alpha * X[i];
        }

        for (int i = m; i + 3 < N; i += 4) {
            Y[i] += alpha * X[i];
            Y[i + 1] += alpha * X[i + 1];
            Y[i + 2] += alpha * X[i + 2];
            Y[i + 3] += alpha * X[i + 3];
        }
    }
    else
    {
        int ix = OFFSET(N, incX);
        int iy = OFFSET(N, incY);

        for (int i = 0; i < N; i++) {
            Y[iy] += alpha * X[ix];
            ix += incX;
            iy += incY;
        }
    }
}

void cblas_cswap(const int N, void *X, const int incX, 
                 void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const float tmp_real = REAL_FLOAT(X, ix);
        const float tmp_imag = IMAG_FLOAT(X, ix);
        REAL_FLOAT(X, ix) = REAL_FLOAT(Y, iy);
        IMAG_FLOAT(X, ix) = IMAG_FLOAT(Y, iy);
        REAL_FLOAT(Y, iy) = tmp_real;
        IMAG_FLOAT(Y, iy) = tmp_imag;
        ix += incX;
        iy += incY;
    }
}


void cblas_ccopy(const int N, const void *X, const int incX, 
                 void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        REAL_FLOAT(Y, iy) = CONST_REAL_FLOAT(X, ix);
        IMAG_FLOAT(Y, iy) = CONST_IMAG_FLOAT(X, ix);
        ix += incX;
        iy += incY;
    }
}

void cblas_caxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (fabs(alpha_real) == 0 && fabs(alpha_imag) == 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        const float x_real = CONST_REAL_FLOAT(X, ix);
        const float x_imag = CONST_IMAG_FLOAT(X, ix);
        REAL_FLOAT(Y, iy) += (alpha_real * x_real - alpha_imag * x_imag);
        IMAG_FLOAT(Y, iy) += (alpha_real * x_imag + alpha_imag * x_real);
        ix += incX;
        iy += incY;
    }
}

void cblas_zswap(const int N, void *X, const int incX, 
                 void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        const double tmp_real = REAL_DOUBLE(X, ix);
        const double tmp_imag = IMAG_DOUBLE(X, ix);
        REAL_DOUBLE(X, ix) = REAL_DOUBLE(Y, iy);
        IMAG_DOUBLE(X, ix) = IMAG_DOUBLE(Y, iy);
        REAL_DOUBLE(Y, iy) = tmp_real;
        IMAG_DOUBLE(Y, iy) = tmp_imag;
        ix += incX;
        iy += incY;
    }
}

void cblas_zcopy(const int N, const void *X, const int incX, 
                 void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (int i = 0; i < N; i++) {
        REAL_DOUBLE(Y, iy) = CONST_REAL_DOUBLE(X, ix);
        IMAG_DOUBLE(Y, iy) = CONST_IMAG_DOUBLE(X, ix);
        ix += incX;
        iy += incY;
    }
}


void cblas_zaxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (fabs(alpha_real) == 0 && fabs(alpha_imag) == 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        const double x_real = CONST_REAL_DOUBLE(X, ix);
        const double x_imag = CONST_IMAG_DOUBLE(X, ix);
        REAL_DOUBLE(Y, iy) += (alpha_real * x_real - alpha_imag * x_imag);
        IMAG_DOUBLE(Y, iy) += (alpha_real * x_imag + alpha_imag * x_real);
        ix += incX;
        iy += incY;
    }
}


void cblas_srotg(float *a, float *b, float *c, float *s)
{
    const float roe = (fabs(*a) > fabs(*b) ? *a : *b);
    const float scale = fabs(*a) + fabs(*b);
    float r, z;

    if (scale != 0.0) {
        const float aos = *a / scale;
        const float bos = *b / scale;
        r = scale * sqrt(aos * aos + bos * bos);
        r = NA_SIGN(roe) * r;
        *c = *a / r;
        *s = *b / r;
        z = 1.0;
        if (fabs(*a) > fabs(*b))
            z = *s;
        if (fabs(*b) >= fabs(*a) && *c != 0.0)
            z = 1.0 / (*c);
    } else {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    }

    *a = r;
    *b = z;
}

void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P)
{
    const float G = 4096.0;
    const float G2 = G * G;
    float D1 = *d1, D2 = *d2, x = *b1, y = b2;
    float h11, h12, h21, h22, u;
    float c, s;

    /* case of d1 < 0, appendix A, second to last paragraph */
    if (D1 < 0.0) {
        P[0] = -1;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = 0;
        *d1 = 0;
        *d2 = 0;
        *b1 = 0;
        return;
    }

    if (D2 * y == 0.0) {
        P[0] = -2;                  /* case of H = I */
        return;
    }

    c = fabs(D1 * x * x);
    s = fabs(D2 * y * y);

    if (c > s) {
        /* case of equation A6 */
        P[0] = 0.0;
        h11 = 1;
        h12 = (D2 * y) / (D1 * x);
        h21 = -y / x;
        h22 = 1;
        u = 1 - h21 * h12;
        if (u <= 0.0) {             /* the case u <= 0 is rejected */
            P[0] = -1;
            P[1] = 0;
            P[2] = 0;
            P[3] = 0;
            P[4] = 0;
            *d1 = 0;
            *d2 = 0;
            *b1 = 0;
            return;
        }

        D1 /= u;
        D2 /= u;
        x *= u;
    } else {
        /* case of equation A7 */
        if (D2 * y * y < 0.0) {
            P[0] = -1;
            P[1] = 0;
            P[2] = 0;
            P[3] = 0;
            P[4] = 0;
            *d1 = 0;
            *d2 = 0;
            *b1 = 0;
            return;
        }

        P[0] = 1;

        h11 = (D1 * x) / (D2 * y);
        h12 = 1;
        h21 = -1;
        h22 = x / y;
        u = 1 + h11 * h22;
        D1 /= u;
        D2 /= u;
        {
            float tmp = D2;
            D2 = D1;
            D1 = tmp;
        }
        x = y * u;
    }

    /* rescale D1 to range [1/G2,G2] */
    while (D1 <= 1.0 / G2 && D1 != 0.0) {
        P[0] = -1;
        D1 *= G2;
        x /= G;
        h11 /= G;
        h12 /= G;
    }

    while (D1 >= G2) {
        P[0] = -1;
        D1 /= G2;
        x *= G;
        h11 *= G;
        h12 *= G;
    }

    /* rescale D2 to range [1/G2,G2] */
    while (fabs(D2) <= 1.0 / G2 && D2 != 0.0) {
        P[0] = -1;
        D2 *= G2;
        h21 /= G;
        h22 /= G;
    }

    while (fabs(D2) >= G2) {
        P[0] = -1;
        D2 /= G2;
        h21 *= G;
        h22 *= G;
    }

    *d1 = D1;
    *d2 = D2;
    *b1 = x;

    if (P[0] == -1.0) {
        P[1] = h11;
        P[2] = h21;
        P[3] = h12;
        P[4] = h22;
    } else if (P[0] == 0.0) {
        P[2] = h21;
        P[3] = h12;
    } else if (P[0] == 1.0) {
        P[1] = h11;
        P[4] = h22;
    }
}

void cblas_srot(const int N, float *X, const int incX,
                float *Y, const int incY, const float c, const float s)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (int i = 0; i < N; i++) {
        const float x = X[ix];
        const float y = Y[iy];
        X[ix] = c * x + s * y;
        Y[iy] = -s * x + c * y;
        ix += incX;
        iy += incY;
    }
}

void cblas_srotm(const int N, float *X, const int incX,
                float *Y, const int incY, const float *P)
{
    int i = OFFSET(N, incX);
    int j = OFFSET(N, incY);
    float h11, h21, h12, h22;

    if (P[0] == -1.0) {
        h11 = P[1];
        h21 = P[2];
        h12 = P[3];
        h22 = P[4];
    } else if (P[0] == 0.0) {
        h11 = 1.0;
        h21 = P[2];
        h12 = P[3];
        h22 = 1.0;
    } else if (P[0] == 1.0) {
        h11 = P[1];
        h21 = -1.0;
        h12 = 1.0;
        h22 = P[4];
    } else if (P[0] == -2.0) {
        return;
    } else {
        fprintf(stderr, "unrecognized value of P[0]\n");
        return;
    }

    for (int n = 0; n < N; n++) {
        const float w = X[i];
        const float z = Y[j];
        X[i] = h11 * w + h12 * z;
        Y[j] = h21 * w + h22 * z;
        i += incX;
        j += incY;
    }
}

void cblas_drotg(double *a, double *b, double *c, double *s)
{
    const double roe = (fabs(*a) > fabs(*b) ? *a : *b);
    const double scale = fabs(*a) + fabs(*b);
    double r, z;

    if (scale != 0.0) {
        const double aos = *a / scale;
        const double bos = *b / scale;
        r = scale * sqrt(aos * aos + bos * bos);
        r = NA_SIGN(roe) * r;
        *c = *a / r;
        *s = *b / r;
        z = 1.0;
        if (fabs(*a) > fabs(*b))
            z = *s;
        if (fabs(*b) >= fabs(*a) && *c != 0.0)
            z = 1.0 / (*c);
    } else {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    }

    *a = r;
    *b = z;
}

void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P)
{
    const double G = 4096.0;
    const double G2 = G * G;
    double D1 = *d1, D2 = *d2, x = *b1, y = b2;
    double h11, h12, h21, h22, u;
    double c, s;

    /* case of d1 < 0, appendix A, second to last paragraph */
    if (D1 < 0.0) {
        P[0] = -1;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = 0;
        *d1 = 0;
        *d2 = 0;
        *b1 = 0;
        return;
    }

    if (D2 * y == 0.0) {
        P[0] = -2;                  /* case of H = I */
        return;
    }

    c = fabs(D1 * x * x);
    s = fabs(D2 * y * y);

    if (c > s) {
        /* case of equation A6 */
        P[0] = 0.0;
        h11 = 1;
        h12 = (D2 * y) / (D1 * x);
        h21 = -y / x;
        h22 = 1;
        u = 1 - h21 * h12;
        if (u <= 0.0) {             /* the case u <= 0 is rejected */
            P[0] = -1;
            P[1] = 0;
            P[2] = 0;
            P[3] = 0;
            P[4] = 0;
            *d1 = 0;
            *d2 = 0;
            *b1 = 0;
            return;
        }

        D1 /= u;
        D2 /= u;
        x *= u;
    } else {
        /* case of equation A7 */
        if (D2 * y * y < 0.0) {
            P[0] = -1;
            P[1] = 0;
            P[2] = 0;
            P[3] = 0;
            P[4] = 0;
            *d1 = 0;
            *d2 = 0;
            *b1 = 0;
            return;
        }

        P[0] = 1;

        h11 = (D1 * x) / (D2 * y);
        h12 = 1;
        h21 = -1;
        h22 = x / y;
        u = 1 + h11 * h22;
        D1 /= u;
        D2 /= u;
        {
            double tmp = D2;
            D2 = D1;
            D1 = tmp;
        }
        x = y * u;
    }

    /* rescale D1 to range [1/G2,G2] */
    while (D1 <= 1.0 / G2 && D1 != 0.0) {
        P[0] = -1;
        D1 *= G2;
        x /= G;
        h11 /= G;
        h12 /= G;
    }

    while (D1 >= G2) {
        P[0] = -1;
        D1 /= G2;
        x *= G;
        h11 *= G;
        h12 *= G;
    }

    /* rescale D2 to range [1/G2,G2] */
    while (fabs(D2) <= 1.0 / G2 && D2 != 0.0) {
        P[0] = -1;
        D2 *= G2;
        h21 /= G;
        h22 /= G;
    }

    while (fabs(D2) >= G2) {
        P[0] = -1;
        D2 /= G2;
        h21 *= G;
        h22 *= G;
    }

    *d1 = D1;
    *d2 = D2;
    *b1 = x;

    if (P[0] == -1.0) {
        P[1] = h11;
        P[2] = h21;
        P[3] = h12;
        P[4] = h22;
    } else if (P[0] == 0.0) {
        P[2] = h21;
        P[3] = h12;
    } else if (P[0] == 1.0) {
        P[1] = h11;
        P[4] = h22;
    }
}

void cblas_drot(const int N, double *X, const int incX,
                double *Y, const int incY, const double c, const double  s)
{
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);
    for (int i = 0; i < N; i++) {
        const double x = X[ix];
        const double y = Y[iy];
        X[ix] = c * x + s * y;
        Y[iy] = -s * x + c * y;
        ix += incX;
        iy += incY;
    }
}

void cblas_drotm(const int N, double *X, const int incX,
                double *Y, const int incY, const double *P)
{
    int i = OFFSET(N, incX);
    int j = OFFSET(N, incY);
    double h11, h21, h12, h22;

    if (P[0] == -1.0) {
        h11 = P[1];
        h21 = P[2];
        h12 = P[3];
        h22 = P[4];
    } else if (P[0] == 0.0) {
        h11 = 1.0;
        h21 = P[2];
        h12 = P[3];
        h22 = 1.0;
    } else if (P[0] == 1.0) {
        h11 = P[1];
        h21 = -1.0;
        h12 = 1.0;
        h22 = P[4];
    } else if (P[0] == -2.0) {
        return;
    } else {
        fprintf(stderr, "unrecognized value of P[0]\n");
        return;
    }

    for (int n = 0; n < N; n++) {
        const double w = X[i];
        const double z = Y[j];
        X[i] = h11 * w + h12 * z;
        Y[j] = h21 * w + h22 * z;
        i += incX;
        j += incY;
    }
}

// x := alpha * x
void cblas_sscal(const int N, const float alpha, float *X, const int incX)
{
    int ix = 0;

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        X[ix] *= alpha;
        ix += incX;
    }
}

void cblas_dscal(const int N, const double alpha, double *X, const int incX)
{
    int ix = 0;

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        X[ix] *= alpha;
        ix += incX;
    }
}

void cblas_cscal(const int N, const void *alpha, void *X, const int incX)
{
    int ix = 0;
    const float alpha_real = CONST_REAL0_FLOAT(alpha);
    const float alpha_imag = CONST_IMAG0_FLOAT(alpha);

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        const float x_real = REAL_FLOAT(X, ix);
        const float x_imag = IMAG_FLOAT(X, ix);
        REAL_FLOAT(X, ix) = x_real * alpha_real - x_imag * alpha_imag;
        IMAG_FLOAT(X, ix) = x_real * alpha_imag + x_imag * alpha_real;
        ix += incX;
    }
}

void cblas_zscal(const int N, const void *alpha, void *X, const int incX)
{
    int ix = 0;
    const double alpha_real = CONST_REAL0_DOUBLE(alpha);
    const double alpha_imag = CONST_IMAG0_DOUBLE(alpha);

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        const double x_real = REAL_DOUBLE(X, ix);
        const double x_imag = IMAG_DOUBLE(X, ix);
        REAL_DOUBLE(X, ix) = x_real * alpha_real - x_imag * alpha_imag;
        IMAG_DOUBLE(X, ix) = x_real * alpha_imag + x_imag * alpha_real;
        ix += incX;
    }
}

void cblas_csscal(const int N, const float alpha, void *X, const int incX)
{
    int ix = 0;

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        REAL_FLOAT(X, ix) *= alpha;
        IMAG_FLOAT(X, ix) *= alpha;
        ix += incX;
    }
}

void cblas_zdscal(const int N, const double alpha, void *X, const int incX)
{
    int ix = 0;

    if (incX <= 0) {
        return;
    }

    for (int i = 0; i < N; i++) {
        REAL_DOUBLE(X, ix) *= alpha;
        IMAG_DOUBLE(X, ix) *= alpha;
        ix += incX;
    }
}

