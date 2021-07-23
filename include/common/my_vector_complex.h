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

#ifndef _MY_VECTOR_COMPLEX_H_
#define _MY_VECTOR_COMPLEX_H_           1

#include <iostream>
#include <cstdio>
#include "complex_t.h"

template<typename T>
class VectorComplex
{
public:
    complex_t<T> *p_;
    int dim_;
    int8_t ref_;   // 0 or 1 : does this own its own memory space
public:
    VectorComplex()
    {
        p_ = NULL;
        dim_ = 0;
        ref_ = 0;
    }

    VectorComplex(int n)
    {
        p_ = new complex_t<T>[n];
        if (p_ = NULL) {
            std::cerr << "Error: NULL pointer in VectorComplex(int) constructor " << "\n";
            std::cerr << "       Most likely out of memory... " << "\n";
            exit(-1);
        }
        dim_ = n;
        ref_ = 0;
    }

    VectorComplex(int n, const complex_t<T>& v)
    {
        p_ = new complex_t<T>[n];
        if (p_ = NULL) {
            std::cerr << "Error: NULL pointer in VectorComplex(int) constructor " << "\n";
            std::cerr << "       Most likely out of memory... " << "\n";
            exit(-1);
        }
        dim_ = n;
        ref_ = 0;
        for (int i = 0; i < n; i++) {
            p_[i] = v;
        }
    }

    complex_t<T>& operator()(int i) { return p_[i]; }
    const complex_t<T>& operator()(int i) const { return p_[i]; }
    complex_t<T>& operator[](int i) { return p_[i]; }
    const complex_t<T>& operator[](int i) { return p_[i]; }

    inline int size() const { return dim_; }
    inline int dim() const  { return dim_; }
    inline int ref() const  { return ref_; }
    inline int null() const { return dim_ == 0; }

    VectorComplex<T>& newsize(int n)
    {
        if (ref_)
        {
            std::cerr << "VectorComplex newsize can't operator on references.\n";
            exit(-1);
        }
        else
        {
            if (dim_ != n)
            {
                // only delete and new if the size of memory is really changing
                if (p_) delete []p_;
                p_ = new complex_t<T>[n];
                if (p_ == NULL) {
                    std::cerr << "Error: NULL pointer in VectorComplex(int) newsize " << "\n";
                    std::cerr << "       Most likely out of memory... " << "\n";
                    exit(-1);
                }

                dim_ = n;
            }
        }

        return *this;
    }

    VectorComplex<T>& operator=(const VectorComplex<T>& m)
    {
        int n = m.dim_;
        int i;
        if (ref_)
        {
            if (dim_ != m.dim_)
            {
                std::cerr << "VectorComplex::operator=  non-conformant assignment.\n";
                exit(-1);
            }

            // handle overlapping matrix references
            if ((m.p_ + m.dim_) >= p_)
            {
                // overlap case, copy backwards to avoid overwriting results
                for (i = n - 1; i >= 0; i--) {
                    p_[i] = m.p_[i];
                }
            }
            else
            {
                for (i=0; i < n; i++) {
                    p_[i] = m.p_[i];
                }
            }
        }
        else
        {
            newsize(n);
            for (i=0; i < n; i++)
            {
                p_[i] = m.p_[i];
            }
        }

        return *this;
    }

    VectorComplex<T>& operator=(const complex_t<T>& m)
    {
        int n = size();
        int nminus4 = n - 4;
        int i;
        // unroll loops to depth of length 4
        for (i = 0; i < nminus4;)
        {
            p_[i++] = m;
            p_[i++] = m;
            p_[i++] = m;
            p_[i++] = m;
        }

        for (; i < n;)
        {
            p_[i++] = m;
        }

        return *this;
    }

};

#endif
