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
#ifndef _SIMD_ARRAY_H_
#define _SIMD_ARRAY_H_      1

#include <new>
#include "my_simd.h"

template<typename T>
struct simd_array
{
    T *p;
#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    simd_array() noexcept : p(0)  {}
#else
    simd_array() : p(0)  {}
#endif
    simd_array(int n) : p((T*) my_simd_malloc(n*sizeof(T)))
    {
        if (p == 0) throw std::bad_alloc();
    }

    ~simd_array() { if (p) my_simd_free(p); }

    void setup(int n)
    {
        if (p) my_simd_free(p);
        p = (T*) my_simd_malloc(n*sizeof(T));
        if (p == 0) throw std::bad_alloc();
    }

    void destroy() { if (p) my_simd_free(p); p = 0; }

#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
    T& operator[](int i) noexcept { return p[i]; }
    const T& operator[](int i) const noexcept { return p[i]; }
    T* operator&() const noexcept { return p; }
#else
    T& operator[](int i) { return p[i]; }
    const T& operator[](int i) const { return p[i]; }
    T* operator&() const { return p; }
#endif
};

#endif
