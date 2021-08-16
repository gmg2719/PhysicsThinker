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

#ifndef SIGNALS_DIMSCAL_H_
#define SIGNALS_DIMSCAL_H_

#include <iostream>

template<typename T> class Dimscal;
template<typename T> std::ostream& operator<<(std::ostream &s, const Dimscal<T> &M);

template <typename T>
class Dimscal
{
protected:
    long dim1_;
    long dim2_;
    long dim3_;
    long w1, w2;
    T *p_;
public:
    Dimscal() { dim1_= dim2_ = dim3_ = 0; w1 = w2 = 0; p_ = NULL; }
    ~Dimscal() { if (p_ != NULL)  delete []p_; }
    Dimscal(long, long, long);
    inline const T& operator() (long dim1, long dim2,
            long dim3) const {
        return p_[dim1*w1+dim2*w2+dim3];
    }
    inline T& operator() (long dim1, long dim2, long dim3) {
        return p_[dim1*w1+dim2*w2+dim3];
    }
    inline T *ptr() { return p_; }
    inline long dim1() { return dim1_; }
    inline long dim2() { return dim2_; }
    inline long dim3() { return dim3_; }
    inline long dim() { return dim1_ * dim2_ * dim3_; }
    inline void mul(T num) {
        for (long i=0; i<dim(); i++) {
            p_[i] *= num;
        }
    }

    Dimscal<T> & operator=(const Dimscal<T>&);

    // common functions
    Dimscal<T> & newsize(long, long, long);
};

template <typename T>
Dimscal<T>::Dimscal(long m, long n, long k)
{
    if (m<=0 || n<=0 || k<=0) {
        p_ = NULL;
        dim1_ = dim2_ = dim3_ = 0;
        std::cerr << "Error: bad value in Dimscal constructor " << std::endl;
        return;
    }
    dim1_ = m;
    dim2_ = n;
    dim3_ = k;
    w1 = n*k;
    w2 = k;
    p_ = new T[m*n*k];

    if (p_ == NULL) {
        std::cerr << "Error: NULL pointer in Dimscal<T> constructor " << std::endl;
        std::cerr << "       Most likely out of memory... " << std::endl;
        exit(-1);
    }
}

template <typename T>
Dimscal<T>& Dimscal<T>::newsize(long m, long n, long k)
{
    if (m<=0 || n<=0 || k<=0) {
        if (p_ != NULL) {
            delete[] p_;
        }
        p_ = NULL;
        dim1_ = dim2_ = dim3_ = 0;
        w1 = w2 = 0;
        std::cerr << "Error: bad value in Dimscal newsize " << std::endl;
        return *this;
    }

    if (p_ != NULL) {
        delete[] p_;
    }
    dim1_ = m;
    dim2_ = n;
    dim3_ = k;
    w1 = n*k;
    w2 = k;
    p_ = new T[m*n*k];
    if (p_ == NULL) {
        p_ = NULL;
        dim1_ = dim2_ = dim3_  = 0;
        w1 = w2 = 0;
        std::cerr << "Error: bad alloc in Dimscal newsize " << std::endl;
        return *this;
    }

    return *this;
}

template <typename T>
Dimscal<T>& Dimscal<T>::operator=(const Dimscal<T>& M)
{
    for (long i=0; i<dim(); i++) {
        p_[i] = M.p_[i];
    }
}

#endif /* SIGNALS_DIMSCAL_H_ */
