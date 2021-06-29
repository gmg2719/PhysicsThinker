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
#ifndef _COMPLEX_T_H_
#define _COMPLEX_T_H_       1

/*********************************************************
 *   Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
 *   Released under the MIT license
 *   http://opensource.org/licenses/mit-license.php
 *********************************************************/
//
// Implement and modify based on the OTFFT FFT library
//

#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    #define NOEXCEPT noexcept
#else
    #define NOEXCEPT
#endif

#if __GNUC__ >= 3
    #define force_inline __attribute__((const, always_inline))
    #define force_inline2 __attribute__((pure, always_inline))
    #define force_inline3 __attribute__((always_inline))
#else
    #define force_inline
    #define force_inline2
    #define force_inline3
#endif

#include <cmath>
#include <complex>

// T : double or float, used for storing different precision floating-point numbers
template<typename T>
struct complex_t
{
    T Re, Im;
    complex_t() NOEXCEPT : Re(0), Im(0)  {}
    complex_t(const T &x) NOEXCEPT : Re(x), Im(0)  {}
    complex_t(const T &x, const T &y) NOEXCEPT : Re(x), Im(y)  {}
    complex_t(const complex_t& z) NOEXCEPT : Re(z.Re), Im(z.Im)  {}
    complex_t(const std::complex<T>& z) NOEXCEPT : Re(z.real()), Im(z.imag())  {}
    operator std::complex<T>() const { return std::complex<T>(Re, Im); }

    complex_t<T>& operator=(const complex_t<T>& z) NOEXCEPT
    {
        Re = z.Re;
        Im = z.Im;
        return *this;
    }

    complex_t<T>& operator+=(const complex_t<T>& z) NOEXCEPT
    {
        Re += z.Re;
        Im += z.Im;
        return *this;
    }

    complex_t<T>& operator-=(const complex_t<T>& z) NOEXCEPT
    {
        Re -= z.Re;
        Im -= z.Im;
        return *this;
    }

    complex_t<T>& operator*=(const T& x) NOEXCEPT
    {
        Re *= x;
        Im *= x;
        return *this;
    }

    complex_t<T>& operator/=(const T& x) NOEXCEPT
    {
        Re /= x;
        Im /= x;
        return *this;
    }

    complex_t<T>& operator*=(const complex_t<T>& z) NOEXCEPT
    {
        const T tmp = Re * z.Re - Im * z.Im;
        Im = Re*z.Im + Im * z.Re;
        Re = tmp;
        return *this;
    }
};

typedef float* __restrict const float_vector;
typedef const float* __restrict const const_float_vector;
typedef complex_t<float>* __restrict const complex_f_vector;
typedef const complex_t<float>* __restrict const const_complex_f_vector;

typedef double* __restrict const double_vector;
typedef const double* __restrict const const_double_vector;
typedef complex_t<double>* __restrict const complex_d_vector;
typedef const complex_t<double>* __restrict const const_complex_d_vector;

// T='float' type operations
static inline float Re(const complex_t<float>& z) NOEXCEPT force_inline;
static inline float Re(const complex_t<float>& z) NOEXCEPT { return z.Re; }
static inline float Im(const complex_t<float>& z) NOEXCEPT force_inline;
static inline float Im(const complex_t<float>& z) NOEXCEPT { return z.Im; }

static inline float norm(const complex_t<float>& z) NOEXCEPT force_inline;
static inline float norm(const complex_t<float>& z) NOEXCEPT
{
    return z.Re*z.Re + z.Im*z.Im;
}
static inline complex_t<float> conj(const complex_t<float>& z) NOEXCEPT force_inline;
static inline complex_t<float> conj(const complex_t<float>& z) NOEXCEPT
{
    return complex_t<float>(z.Re, -z.Im);
}
static inline complex_t<float> jx(const complex_t<float>& z) NOEXCEPT force_inline;
static inline complex_t<float> jx(const complex_t<float>& z) NOEXCEPT
{
    return complex_t<float>(-z.Im, z.Re);
}
static inline complex_t<float> neg(const complex_t<float>& z) NOEXCEPT force_inline;
static inline complex_t<float> neg(const complex_t<float>& z) NOEXCEPT
{
    return complex_t<float>(-z.Re, -z.Im);
}
static inline complex_t<float> mjx(const complex_t<float>& z) NOEXCEPT force_inline;
static inline complex_t<float> mjx(const complex_t<float>& z) NOEXCEPT
{
    return complex_t<float>(z.Im, -z.Re);
}

static inline complex_t<float> operator+(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT force_inline;
static inline complex_t<float> operator+(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT
{
    return complex_t<float>(a.Re + b.Re, a.Im + b.Im);
}
static inline complex_t<float> operator-(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT force_inline;
static inline complex_t<float> operator-(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT
{
    return complex_t<float>(a.Re - b.Re, a.Im - b.Im);
}
static inline complex_t<float> operator*(const float& a, const complex_t<float>& b) NOEXCEPT force_inline;
static inline complex_t<float> operator*(const float& a, const complex_t<float>& b) NOEXCEPT
{
    return complex_t<float>(a*b.Re, a*b.Im);
}
static inline complex_t<float> operator*(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT force_inline;
static inline complex_t<float> operator*(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT
{
    return complex_t<float>(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
}
static inline complex_t<float> operator/(const complex_t<float>& a, const float& b) NOEXCEPT force_inline;
static inline complex_t<float> operator/(const complex_t<float>& a, const float& b) NOEXCEPT
{
    return complex_t<float>(a.Re/b, a.Im/b);
}
static inline complex_t<float> operator/(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT force_inline;
static inline complex_t<float> operator/(const complex_t<float>& a, const complex_t<float>& b) NOEXCEPT
{
    const float b2 = b.Re*b.Re + b.Im*b.Im;
    return (a * conj(b)) / b2;
}

static inline complex_t<float> expj(const float& theta) NOEXCEPT force_inline;
static inline complex_t<float> expj(const float& theta) NOEXCEPT
{
    const float one = 1.0;
    return complex_t<float>(std::polar(one, theta));
}

// T='double' type operations
static inline double Re(const complex_t<double>& z) NOEXCEPT force_inline;
static inline double Re(const complex_t<double>& z) NOEXCEPT { return z.Re; }
static inline double Im(const complex_t<double>& z) NOEXCEPT force_inline;
static inline double Im(const complex_t<double>& z) NOEXCEPT { return z.Im; }

static inline double norm(const complex_t<double>& z) NOEXCEPT force_inline;
static inline double norm(const complex_t<double>& z) NOEXCEPT
{
    return z.Re*z.Re + z.Im*z.Im;
}
static inline complex_t<double> conj(const complex_t<double>& z) NOEXCEPT force_inline;
static inline complex_t<double> conj(const complex_t<double>& z) NOEXCEPT
{
    return complex_t<double>(z.Re, -z.Im);
}
static inline complex_t<double> jx(const complex_t<double>& z) NOEXCEPT force_inline;
static inline complex_t<double> jx(const complex_t<double>& z) NOEXCEPT
{
    return complex_t<double>(-z.Im, z.Re);
}
static inline complex_t<double> neg(const complex_t<double>& z) NOEXCEPT force_inline;
static inline complex_t<double> neg(const complex_t<double>& z) NOEXCEPT
{
    return complex_t<double>(-z.Re, -z.Im);
}
static inline complex_t<double> mjx(const complex_t<double>& z) NOEXCEPT force_inline;
static inline complex_t<double> mjx(const complex_t<double>& z) NOEXCEPT
{
    return complex_t<double>(z.Im, -z.Re);
}

static inline complex_t<double> operator+(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT force_inline;
static inline complex_t<double> operator+(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT
{
    return complex_t<double>(a.Re + b.Re, a.Im + b.Im);
}
static inline complex_t<double> operator-(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT force_inline;
static inline complex_t<double> operator-(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT
{
    return complex_t<double>(a.Re - b.Re, a.Im - b.Im);
}
static inline complex_t<double> operator*(const double& a, const complex_t<double>& b) NOEXCEPT force_inline;
static inline complex_t<double> operator*(const double& a, const complex_t<double>& b) NOEXCEPT
{
    return complex_t<double>(a*b.Re, a*b.Im);
}
static inline complex_t<double> operator*(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT force_inline;
static inline complex_t<double> operator*(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT
{
    return complex_t<double>(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
}
static inline complex_t<double> operator/(const complex_t<double>& a, const double& b) NOEXCEPT force_inline;
static inline complex_t<double> operator/(const complex_t<double>& a, const double& b) NOEXCEPT
{
    return complex_t<double>(a.Re/b, a.Im/b);
}
static inline complex_t<double> operator/(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT force_inline;
static inline complex_t<double> operator/(const complex_t<double>& a, const complex_t<double>& b) NOEXCEPT
{
    const double b2 = b.Re*b.Re + b.Im*b.Im;
    return (a * conj(b)) / b2;
}

static inline complex_t<double> expj(const double& theta) NOEXCEPT force_inline;
static inline complex_t<double> expj(const double& theta) NOEXCEPT
{
    const double one = 1.0;
    return complex_t<double>(std::polar(one, theta));
}

#endif
