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

#ifndef _PAIRWISE_SUM_H_
#define _PAIRWISE_SUM_H_      1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

#include <cmath>

/**
 * @file pairwise_sum.h
 * @brief Utility function for the accurate pairwise sum of a list of floating
 *        point numbers.
 * @author William Boyd (wboyd@mit.edu)
 * @date June 13, 2013
 */

/**
 * @brief Performs a pairwise sum of an array of numbers.
 * @details This type of summation uses a divide-and-conquer algorithm which
 *          is necessary to bound the error for summations of large sequences
 *          of numbers.
 * @param vector an array of numbers
 * @param length the length of the array
 * @return the sum of all numbers in the array
 */
template <typename T>
inline T pairwise_sum(T* vector, int length)
{
  T sum = 0;

  /* Base case: if length is less than 16, perform summation */
  if (length < 16) {

    #pragma simd reduction(+:sum)
    for (int i=0; i < length; i++)
      sum += vector[i];
  }

  else {
    int offset = length % 2;
    length = floor(length / 2);
    sum = pairwise_sum<T>(&vector[0], length) +
          pairwise_sum<T>(&vector[length], length+offset);
  }

  return sum;
}

#endif
