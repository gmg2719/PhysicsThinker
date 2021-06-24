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
#ifndef _MY_DEBUG_HPP_
#define _MY_DEBUG_HPP_      1

#include <iostream>
#include <iomanip>
#include <fstream>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cmath>

template<typename T>
bool debug_cmp_array(int size, T *src, T *dst, double epsilon=1E-6)
{
    double tmp;

    // Compare two arrays, and ensure the same results !
    for (int i = 0; i < size; i++) {
        tmp = src[i] - dst[i];
        if (fabs(tmp) > epsilon) {
            return false;
        }
    }

    return true;
}

template<typename T>
void debug_print_array(const char * name, int size, T *ans)
{
    std::cout << name << " : " << std::endl;
    for (int i = 0; i < size; i++) {
#if __cplusplus < 201103L
        std::cout << ans[i] << " ";
#else
        if (std::is_floating_point<T>::value == 1) {
            // Output the floating point numbers
            std::cout << std::scientific << std::setprecision(5) << ans[i] << " ";
        } else {
            std::cout << ans[i] << " ";
        }
#endif
        if ((i+1) % 8 == 0)  std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void debug_file_array(const char * filename, int size, T *ans)
{
    std::ofstream file;
    file.open(filename);
    for (int i = 0; i < size; i++) {
#if __cplusplus < 201103L
        file << ans[i] << " ";
#else
        if (std::is_floating_point<T>::value == 1) {
            // Output the floating point numbers
            file << std::scientific << std::setprecision(6) << ans[i] << " ";
        } else {
            file << ans[i] << " ";
        }
#endif
        if ((i+1) % 8 == 0)  file << std::endl;
    }
    file << std::endl;
    file.close();
}

#endif
