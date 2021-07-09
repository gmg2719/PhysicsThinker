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
#ifndef _MY_TIME_H_
#define _MY_TIME_H_     1

#include <ctime>
#include <sys/time.h>
#ifdef __linux__
    #include <asm/unistd.h>
#endif

#if defined(__GNUC__) || defined(__GNUG__)
static inline uint64_t my_now(void)    __attribute__((always_inline));
#else
static inline uint64_t my_now(void);
#endif
static inline uint64_t my_now(void)
{
#if defined(__x86_64)
    uint32_t t_low, t_high;
    __asm__ __volatile__("rdtsc":"=a"(t_low),"=d"(t_high));

    return (uint64_t)t_high << 32 | t_low;
#elif defined(__aarch64__)
    uint64_t t;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r" (t));
    return t;
#endif
}

#if defined(__GNUC__) || defined(__GNUG__)
static inline uint64_t my_us_gettimeofday(void)    __attribute__((always_inline));
#else
static inline uint64_t my_us_gettimeofday(void);
#endif
static inline uint64_t my_us_gettimeofday(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return t.tv_sec * 1000000ULL + t.tv_usec;
}

#if defined(__GNUC__) || defined(__GNUG__)
static inline uint64_t my_us_clockgettime(void)    __attribute__((always_inline));
#else
static inline uint64_t my_us_clockgettime(void);
#endif
static inline uint64_t my_us_clockgettime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

#endif
