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
#ifndef _MY_MATRIX_H_
#define _MY_MATRIX_H_       1

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

template<typename T, size_t M, size_t N>
class MatrixFixed
{
public:
    T _m[M][N];

public:
    MatrixFixed() = default;

    explicit MatrixFixed(const T x[M*N])
    {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                _m[i][j] = x[N*i + j];
            }
        }
    }

    explicit MatrixFixed(const T x[M][N])
    {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                _m[i][j] = x[i][j];
            }
        }
    }

    MatrixFixed(const MatrixFixed &other)
    {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                _m[i][j] = other(i, j);
            }
        }
    }

    inline const T &operator()(size_t i, size_t j) const
    {
        assert((i < M) && (i >= 0));
        assert((j < N) && (j >= 0));
        return _m[i][j];
    }

    inline T &operator()(size_t i, size_t j)
    {
        assert((i < M) && (i >= 0));
        assert((j < N) && (j >= 0));
        return _m[i][j];
    }

    MatrixFixed<T, M, N> & operator=(const MatrixFixed<T, M, N> &other)
    {
        if (this != &other) {
            MatrixFixed<T, M, N> &p = *this;
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    p[i][j] = other(i, j);
                }
            }
        }

        return *this;
    }

    void copy_to(T dst[M*N]) const
    {
        const MatrixFixed<T, M, N> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                dst[N * i + j] = p(i, j);
            }
        }
    }

    template<size_t P>
    MatrixFixed<T, M, P> operator*(const MatrixFixed<T, N, P> &other) const
    {
        const MatrixFixed<T, M, N> &p = *this;
        MatrixFixed<T, M, P> res;

        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < P; k++) {
                for (size_t j = 0; j < N; j++) {
                    res(i, k) += p(i, j) * other(j, k);
                }
            }
        }

        return res;
    }

    MatrixFixed<T, M, N> mul_each(const MatrixFixed<T, M, N> &other) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j)*other(i, j);
            }
        }

        return res;
    }

    MatrixFixed<T, M, N> div_each(const MatrixFixed<T, M, N> &other) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j)/other(i, j);
            }
        }

        return res;
    }

    MatrixFixed<T, M, N> operator+(const MatrixFixed<T, M, N> &other) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) + other(i, j);
            }
        }

        return res;
    }

    MatrixFixed<T, M, N> operator-(const MatrixFixed<T, M, N> &other) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) - other(i, j);
            }
        }

        return res;
    }

    MatrixFixed<T, M, N> operator-() const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = -p(i, j);
            }
        }

        return res;
    }

    void operator+=(const MatrixFixed<T, M, N> &other)
    {
        MatrixFixed<T, M, N> &self = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                self(i, j) += other(i, j);
            }
        }
    }

    void operator-=(const MatrixFixed<T, M, N> &other)
    {
        MatrixFixed<T, M, N> &self = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                self(i, j) -= other(i, j);
            }
        }
    }

    template<size_t P>
    void operator*=(const MatrixFixed<T, N, P> &other)
    {
        MatrixFixed<T, M, N> &p = *this;
        p = p * other;
    }

    MatrixFixed<T, M, N> operator*(T scalar) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) * scalar;
            }
        }

        return res;
    }

    inline MatrixFixed<T, M, N> operator/(T scalar) const
    {
        return (*this)*(1/scalar);
    }

    MatrixFixed<T, M, N> operator+(T scalar) const
    {
        MatrixFixed<T, M, N> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) + scalar;
            }
        }

        return res;
    }

    inline MatrixFixed<T, M, N> operator-(T scalar) const
    {
        return (*this) + (-1*scalar);
    }

    void operator*=(T scalar)
    {
        MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) *= scalar;
            }
        }
    }

    void operator/=(T scalar)
    {
        MatrixFixed<T, M, N> &p = *this;
        p *= (T(1) / scalar);
    }

    inline void operator+=(T scalar)
    {
        MatrixFixed<T, M, N> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) += scalar;
            }
        }
    }

    inline void operator-=(T scalar)
    {
        MatrixFixed<T, M, N> &p = *this;
        p += (-scalar);
    }

    bool operator==(const MatrixFixed<T, M, N> &other) const
    {
        return isEqual(*this, other);
    }

    bool operator!=(const MatrixFixed<T, M, N> &other) const
    {
        const MatrixFixed<T, M, N> &p = *this;
        return !(p == other);
    }

    void write_string(char * buf, size_t n) const
    {
        if (buf == nullptr)  return;
        buf[0] = '\0';
        const MatrixFixed<T, M, N> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                snprintf(buf + strlen(buf), n - strlen(buf), "\t%8.8g", double(p(i, j)));
            }
            snprintf(buf + strlen(buf), n - strlen(buf), "\n");
        }
    }

    void print() const
    {
        // element: tab, point, 8 digits, 4 scientific notation chars; row: newline; string: \0 end
        static const size_t n = 15*N*M + M + 1;
        char * buf = new char[n];
        write_string(buf, n);
        printf("%s\n", buf);
        delete[] buf;
    }

    MatrixFixed<T, N, M> transpose() const
    {
        MatrixFixed<T, N, M> res;
        const MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(j, i) = p(i, j);
            }
        }

        return res;
    }

    inline MatrixFixed<T, N, M> TR() const
    {
        return transpose();
    }

    void set_zero()
    {
        memset(_m, 0, sizeof(_m));
    }

    inline void zero()
    {
        set_zero();
    }

    void set_all(T val)
    {
        MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) = val;
            }
        }
    }

    inline void set_one()
    {
        set_all(1);
    }

    inline void set_nan()
    {
        set_all(NAN);
    }

    void set_identity()
    {
        set_zero();
        MatrixFixed<T, M, N> &p = *this;

        const size_t min_i = M > N ? N : M;
        for (size_t i = 0; i < min_i; i++) {
            p(i, i) = 1;
        }
    }

    inline void identity()
    {
        set_identity();
    }

    inline void swap_rows(size_t a, size_t b)
    {
        assert(a < M);
        assert(b < M);

        if (a == b) {
            return;
        }

        MatrixFixed<T, M, N> &p = *this;

        for (size_t j = 0; j < N; j++) {
            T tmp = p(a, j);
            p(a, j) = p(b, j);
            p(b, j) = tmp;
        }
    }

    inline void swap_cols(size_t a, size_t b)
    {
        assert(a < N);
        assert(b < N);

        if (a == b) {
            return;
        }

        MatrixFixed<T, M, N> &p = *this;

        for (size_t i = 0; i < M; i++) {
            T tmp = p(i, a);
            p(i, a) = p(i, b);
            p(i, b) = tmp;
        }
    }

    MatrixFixed<T, M, N> abs() const
    {
        MatrixFixed<T, M, N> r;
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                r(i,j) = T(fabs((*this)(i,j)));
            }
        }
        return r;
    }

    T max() const
    {
        T max_val = (*this)(0,0);
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                T val = (*this)(i,j);
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        return max_val;
    }

    T min() const
    {
        T min_val = (*this)(0,0);
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                T val = (*this)(i,j);
                if (val < min_val) {
                    min_val = val;
                }
            }
        }
        return min_val;
    }

    bool isAllNan() const
    {
        const MatrixFixed<float, M, N> &p = *this;
        bool result = true;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                result = result && std::isnan(p(i, j));
            }
        }
        return result;
    }
};

template<typename T>
class Matrix
{
public:
    T *_m;
    size_t M;
    size_t N;

public:
    Matrix()
    {
        M = 0;
        N = 0;
        _m = nullptr;
    }

    explicit Matrix(size_t m, size_t n)
    {
        M = m;
        N = n;
        _m = new T[m * n];
        memset(_m, 0, sizeof(T)*M*N);
    }

    explicit Matrix(size_t m, size_t n, const T *x)
    {
        M = m;
        N = n;
        _m = new T[m * n];
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                _m[i*N + j] = x[i*N + j];
            }
        }
    }

    Matrix(const Matrix<T> &other)
    {
        if (_m != NULL) {
            delete []_m;
        }

        M = other.M;
        N = other.N;
        _m = new T[M * N];
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                _m[i*N + j] = other(i, j);
            }
        }
    }

    inline const T &operator()(size_t i, size_t j) const
    {
        assert((i < M) && (i >= 0));
        assert((j < N) && (j >= 0));
        return _m[i*N + j];
    }

    inline T &operator()(size_t i, size_t j)
    {
        assert((i < M) && (i >= 0));
        assert((j < N) && (j >= 0));
        return _m[i*N + j];
    }

    Matrix<T> & operator=(const Matrix<T> &other)
    {
        if (this != &other) {
            Matrix<T> &p = *this;
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    p(i, j) = other(i, j);
                }
            }
        }

        return *this;
    }

    void resize(size_t m, size_t n)
    {
        if (_m != nullptr) {
            delete []_m;
            M = 0;
            N = 0;
        }

        M = m;
        N = n;
        _m = new T[m * n];
        memset(_m, 0, sizeof(T)*M*N);
    }

    void copy_to(T *dst) const
    {
        const Matrix<T> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                dst[N * i + j] = p(i, j);
            }
        }
    }

    template<size_t P>
    Matrix<T> operator*(const Matrix<T> &other) const
    {
        const Matrix<T> &p = *this;
        Matrix<T> res;

        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < P; k++) {
                for (size_t j = 0; j < N; j++) {
                    res(i, k) += p(i, j) * other(j, k);
                }
            }
        }

        return res;
    }

    Matrix<T> mul_each(const Matrix<T> &other) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j)*other(i, j);
            }
        }

        return res;
    }

    Matrix<T> div_each(const Matrix<T> &other) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j)/other(i, j);
            }
        }

        return res;
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) + other(i, j);
            }
        }

        return res;
    }

    Matrix<T> operator-(const Matrix<T> &other) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) - other(i, j);
            }
        }

        return res;
    }

    Matrix<T> operator-() const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = -p(i, j);
            }
        }

        return res;
    }

    void operator+=(const Matrix<T> &other)
    {
        Matrix<T> &self = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                self(i, j) += other(i, j);
            }
        }
    }

    void operator-=(const Matrix<T> &other)
    {
        Matrix<T> &self = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                self(i, j) -= other(i, j);
            }
        }
    }

    template<size_t P>
    void operator*=(const Matrix<T> &other)
    {
        Matrix<T> &p = *this;
        p = p * other;
    }

    Matrix<T> operator*(T scalar) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) * scalar;
            }
        }

        return res;
    }

    inline Matrix<T> operator/(T scalar) const
    {
        return (*this)*(1/scalar);
    }

    Matrix<T> operator+(T scalar) const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(i, j) = p(i, j) + scalar;
            }
        }

        return res;
    }

    inline Matrix<T> operator-(T scalar) const
    {
        return (*this) + (-1*scalar);
    }

    void operator*=(T scalar)
    {
        Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) *= scalar;
            }
        }
    }

    void operator/=(T scalar)
    {
        Matrix<T> &p = *this;
        p *= (T(1) / scalar);
    }

    inline void operator+=(T scalar)
    {
        Matrix<T> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) += scalar;
            }
        }
    }

    inline void operator-=(T scalar)
    {
        Matrix<T> &p = *this;
        p += (-scalar);
    }

    bool operator==(const Matrix<T> &other) const
    {
        return isEqual(*this, other);
    }

    bool operator!=(const Matrix<T> &other) const
    {
        const Matrix<T> &p = *this;
        return !(p == other);
    }

    void write_string(char * buf, size_t n) const
    {
        if (buf == nullptr)  return;
        buf[0] = '\0';
        const Matrix<T> &p = *this;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                snprintf(buf + strlen(buf), n - strlen(buf), "%10.6g", double(p(i, j)));
            }
            snprintf(buf + strlen(buf), n - strlen(buf), "\n");
        }
    }

    void print() const
    {
        // element: tab, point, 8 digits, 4 scientific notation chars; row: newline; string: \0 end
        static const size_t n = 15*N*M + M + 1;
        char * buf = new char[n];

        printf("matrix size = (%d, %d)\n", M, N);
        write_string(buf, n);
        printf("%s\n", buf);
        delete[] buf;
    }

    Matrix<T> transpose() const
    {
        Matrix<T> res;
        const Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                res(j, i) = p(i, j);
            }
        }

        return res;
    }

    inline Matrix<T> TR() const
    {
        return transpose();
    }

    void inverse()
    {
        fprintf(stderr, "Not supported now !\n");
    }

    void set_zero()
    {
        memset(_m, 0, sizeof(_m));
    }

    inline void zero()
    {
        set_zero();
    }

    void set_all(T val)
    {
        Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                p(i, j) = val;
            }
        }
    }

    inline void set_one()
    {
        set_all(1);
    }

    inline void set_nan()
    {
        set_all(NAN);
    }

    void set_identity()
    {
        set_zero();
        Matrix<T> &p = *this;

        const size_t min_i = M > N ? N : M;
        for (size_t i = 0; i < min_i; i++) {
            p(i, i) = 1;
        }
    }

    inline void identity()
    {
        set_identity();
    }

    inline void swap_rows(size_t a, size_t b)
    {
        assert(a < M);
        assert(b < M);

        if (a == b) {
            return;
        }

        Matrix<T> &p = *this;

        for (size_t j = 0; j < N; j++) {
            T tmp = p(a, j);
            p(a, j) = p(b, j);
            p(b, j) = tmp;
        }
    }

    inline void swap_cols(size_t a, size_t b)
    {
        assert(a < N);
        assert(b < N);

        if (a == b) {
            return;
        }

        Matrix<T> &p = *this;

        for (size_t i = 0; i < M; i++) {
            T tmp = p(i, a);
            p(i, a) = p(i, b);
            p(i, b) = tmp;
        }
    }

    Matrix<T> abs() const
    {
        Matrix<T> r;
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                r(i,j) = T(fabs((*this)(i,j)));
            }
        }
        return r;
    }

    T max() const
    {
        T max_val = (*this)(0,0);
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                T val = (*this)(i,j);
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        return max_val;
    }

    T min() const
    {
        T min_val = (*this)(0,0);
        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<N; j++) {
                T val = (*this)(i,j);
                if (val < min_val) {
                    min_val = val;
                }
            }
        }
        return min_val;
    }

    bool isAllNan() const
    {
        const Matrix<float> &p = *this;
        bool result = true;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                result = result && std::isnan(p(i, j));
            }
        }
        return result;
    }
};

template<typename T>
inline void my_matrix_2x2inv(T& a11, T& a12, T& a21, T& a22)
{
    T det = a11 * a22 - a12 * a21;
    T t11 = a11;
    T t22 = a22;
    a11 = t22 / det;
    a12 = -a12 / det;
    a21 = -a21 / det;
    a22 = t11 / det;
}

template<typename T>
inline void my_matrix_3x3inv(T& f11, T& f12, T& f13, T& f21, T& f22, T& f23,
                             T& f31, T& f32, T& f33)
{
    T det_f = f11*f22*f33+f12*f23*f31+f13*f21*f32-f31*f22*f13-f32*f23*f11-f33*f21*f12;
    T a11 = (f22*f33-f23*f32)/det_f;
    T a12 = (f32*f13-f33*f12)/det_f;
    T a13 = (f12*f23-f13*f22)/det_f;
    T a21 = (f31*f23-f21*f33)/det_f;
    T a22 = (f11*f33-f31*f13)/det_f;
    T a23 = (f21*f13-f11*f23)/det_f;
    T a31 = (f21*f32-f31*f22)/det_f;
    T a32 = (f31*f12-f11*f32)/det_f;
    T a33 = (f11*f22-f21*f12)/det_f;

    f11 = a11;  f12 = a12;  f13 = a13;
    f21 = a21;  f22 = a22;  f23 = a23;
    f31 = a31;  f32 = a32;  f33 = a33;
}

#endif

