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
#ifndef _MY_SIMD_H_
#define _MY_SIMD_H_     1

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
#else
    // ARM-64 neon supported.
    #include <arm_neon.h>
#endif

#include <cstdio>
#include <cstdint>

#ifdef __AVX512__
    typedef __m512      simd_f_t;
    typedef __m512i     simd_i_t;
#else
#ifdef __AVX2__
    typedef __m256      simd_f_t;
    typedef __m256i     simd_i_t;
#else
#if defined(__x86_64__) || defined(__i386__)
    // SSE supported
    typedef __m128      simd_f_t;
    typedef __m128i     simd_i_t;
#else
#ifdef __aarch64__
    // arm64 supported
    typedef float32x4_t simd_f_t;
    typedef int32x4_t   simd_i_t;
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__

//
// Aligned memory allocator
//
#ifdef __AVX512__
    static inline void* my_simd_malloc(const size_t n) { return _mm_malloc(n, 64); }
    static inline void my_simd_free(void *p) { _mm_free(p); }
#else
#ifdef __AVX2__
    static inline void* my_simd_malloc(const size_t n) { return _mm_malloc(n, 32); }
    static inline void my_simd_free(void *p) { _mm_free(p); }
#else
#if defined(__x86_64__) || defined(__i386__)
    // SSE supported
    static inline void* my_simd_malloc(const size_t n) { return _mm_malloc(n, 16); }
    static inline void my_simd_free(void *p) { _mm_free(p); }
#else
#ifdef __aarch64__
    // arm64 supported
    static inline void* my_simd_malloc(const size_t n) { return malloc(n); }
    static inline void my_simd_free(void *p) { free(p); }
#else
    // Others
    static inline void* my_simd_malloc(const size_t n) { return malloc(n); }
    static inline void my_simd_free(void *p) { free(p); }
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__

//////////////////////////////////////
// int32_t simd basic operations    //
//////////////////////////////////////
static inline simd_i_t my_simd_i_load(const int32_t *x)
{
#ifdef __AVX512__
    return _mm512_load_epi32((__m512i *)x);
#else
#ifdef __AVX2__
    return _mm256_load_si256((__m256i *)x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_load_si128((__m128i *)x);
#else
#ifdef __aarch64__
    return vld1q_s32((int *)x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_i_t my_simd_i_loadu(const int32_t *x)
{
#ifdef __AVX512__
    return _mm512_loadu_epi32((__m512i *)x);
#else
#ifdef __AVX2__
    return _mm256_loadu_si256((__m256i *)x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_loadu_si128((__m128i *)x);
#else
#ifdef __aarch64__
    return vld1q_s32((int *)x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline void my_simd_i_store(int32_t *x, simd_i_t reg)
{
#ifdef __AVX512__
    _mm512_store_epi32((__m512i *)x, reg);
#else
#ifdef __AVX2__
    _mm256_store_si256((__m256i *)x, reg);
#else
#if defined(__x86_64__) || defined(__i386__)
    _mm_store_si128((__m128i *)x, reg);
#else
#ifdef __aarch64__
    vst1q_s32((int *)x, reg);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline void my_simd_i_storeu(int32_t *x, simd_i_t reg)
{
#ifdef __AVX512__
    _mm512_storeu_epi32((__m512i *)x, reg);
#else
#ifdef __AVX2__
    _mm256_storeu_si256((__m256i *)x, reg);
#else
#if defined(__x86_64__) || defined(__i386__)
    _mm_storeu_si128((__m128i *)x, reg);
#else
#ifdef __aarch64__
    vst1q_s32((int *)x, reg);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_i_t my_simd_i_set1(int32_t x)
{
#ifdef __AVX512__
    return _mm512_set1_epi32(x);
#else
#ifdef __AVX2__
    return _mm256_set1_epi32(x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_set1_epi32(x);
#else
#ifdef __aarch64__
    return vdupq_n_s32(x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_i_t my_simd_i_add(simd_i_t a, simd_i_t b)
{
#ifdef __AVX512__
    return _mm512_add_epi32(a, b);
#else
#ifdef __AVX2__
    return _mm256_add_epi32(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_add_epi32(a, b);
#else
#ifdef __aarch64__
    return vaddq_s32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_i_t my_simd_i_sub(simd_i_t a, simd_i_t b)
{
#ifdef __AVX512__
    return _mm512_sub_epi32(a, b);
#else
#ifdef __AVX2__
    return _mm256_sub_epi32(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_sub_epi32(a, b);
#else
#ifdef __aarch64__
    return vsubq_s32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_i_t my_simd_i_mul(simd_i_t a, simd_i_t b)
{
#ifdef __AVX512__
    return _mm512_mul_epi32(a, b);
#else
#ifdef __AVX2__
    return _mm256_mul_epi32(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_mul_epi32(a, b);
#else
#ifdef __aarch64__
    return vmulq_s32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

//////////////////////////////////////
// float32_t simd basic operations  //
//////////////////////////////////////
static inline simd_f_t my_simd_f_load(const float *x)
{
#ifdef __AVX512__
    return _mm512_load_ps(x);
#else
#ifdef __AVX2__
    return _mm256_load_ps(x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_load_ps(x);
#else
#ifdef __aarch64__
    return vld1q_f32(x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_loadu(const float *x)
{
#ifdef __AVX512__
    return _mm512_loadu_ps(x);
#else
#ifdef __AVX2__
    return _mm256_loadu_ps(x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_loadu_ps(x);
#else
#ifdef __aarch64__
    return vld1q_f32(x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline void my_simd_f_store(float *x, simd_f_t reg)
{
#ifdef __AVX512__
    _mm512_store_ps(x, reg);
#else
#ifdef __AVX2__
    _mm256_store_ps(x, reg);
#else
#if defined(__x86_64__) || defined(__i386__)
    _mm_store_ps(x, reg);
#else
#ifdef __aarch64__
    vst1q_f32(x, reg);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline void my_simd_f_storeu(float *x, simd_f_t reg)
{
#ifdef __AVX512__
    _mm512_storeu_ps(x, reg);
#else
#ifdef __AVX2__
    _mm256_storeu_ps(x, reg);
#else
#if defined(__x86_64__) || defined(__i386__)
    _mm_storeu_ps(x, reg);
#else
#ifdef __aarch64__
    vst1q_f32(x, reg);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_set1(float x)
{
#ifdef __AVX512__
    return _mm512_set1_ps(x);
#else
#ifdef __AVX2__
    return _mm256_set1_ps(x);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_set1_ps(x);
#else
#ifdef __aarch64__
    return vdupq_n_f32(x);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_zero(void)
{
#ifdef __AVX512__
    return _mm512_setzero_ps();
#else
#ifdef __AVX2__
    return _mm256_setzero_ps();
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_setzero_ps();
#else
#ifdef __aarch64__
    return vdupq_n_f32(0);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_swap(simd_f_t a)
{
#ifdef __AVX512__
    return _mm512_permute_ps(a, 0b10110001);
#else
#ifdef __AVX2__
    return _mm256_permute_ps(a, 0b10110001);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_shuffle_ps(a, a, 0b10110001);
#else
#ifdef __aarch64__
    return vcombine_f32(vrev64_f32(vget_low_f32(a)), vrev64_f32(vget_high_f32(a)));
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_add(simd_f_t a, simd_f_t b)
{
#ifdef __AVX512__
    return _mm512_add_ps(a, b);
#else
#ifdef __AVX2__
    return _mm256_add_ps(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_add_ps(a, b);
#else
#ifdef __aarch64__
    return vaddq_f32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_sub(simd_f_t a, simd_f_t b)
{
#ifdef __AVX512__
    return _mm512_sub_ps(a, b);
#else
#ifdef __AVX2__
    return _mm256_sub_ps(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_sub_ps(a, b);
#else
#ifdef __aarch64__
    return vsubq_f32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_mul(simd_f_t a, simd_f_t b)
{
#ifdef __AVX512__
    return _mm512_mul_ps(a, b);
#else
#ifdef __AVX2__
    return _mm256_mul_ps(a, b);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_mul_ps(a, b);
#else
#ifdef __aarch64__
    return vmulq_f32(a, b);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_rcp(simd_f_t a)
{
#ifdef __AVX512__
    return _mm512_rcp14_ps(a);
#else
#ifdef __AVX2__
    return _mm256_rcp_ps(a);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_rcp_ps(a);
#else
#ifdef __aarch64__
    return vmulq_f32(vrecpeq_f32(a), vrecpsq_f32(vrecpeq_f32(a), a));
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_neg(simd_f_t a)
{
#ifdef __AVX512__
    return _mm512_xor_ps(_mm512_set1_ps(-0.0f), a);
#else
#ifdef __AVX2__
    return _mm256_xor_ps(_mm256_set1_ps(-0.0f), a);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_xor_ps(_mm_set1_ps(-0.0f), a);
#else
#ifdef __aarch64__
    return vnegq_f32(a);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_abs(simd_f_t a)
{
#ifdef __AVX512__
    return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), a);
#else
#ifdef __AVX2__
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
#else
#ifdef __aarch64__
    return vabsq_f32(a);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

static inline simd_f_t my_simd_f_sqrt(simd_f_t a)
{
#ifdef __AVX512__
    return _mm512_sqrt_ps(a);
#else
#ifdef __AVX2__
    return _mm256_sqrt_ps(a);
#else
#if defined(__x86_64__) || defined(__i386__)
    return _mm_sqrt_ps(a);
#else
#ifdef __aarch64__
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(a)；
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    float32x4_t result = vmulq_f32(a, sqrt_reciprocal);
    // Prevent zero in neon if there is NaN
    float32x4_t zeros = vmovq_n_f32(0);
    uint32x4_t mask = vceqq_f32(a, zeros);
    return vbslq_f32(mask, zeros, result);
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__
}

#ifdef __AVX512__
    #define SIMD_F32_WORD_SIZE  16
    #define SIMD_I32_WORD_SIZE  16
    #define SIMD_IS_ALIGNED(ptr)    (((size_t)(ptr)&0x3F) == 0)
#else
#ifdef __AVX2__
    #define SIMD_F32_WORD_SIZE  8
    #define SIMD_I32_WORD_SIZE  8
    #define SIMD_IS_ALIGNED(ptr)    (((size_t)(ptr)&0x1F) == 0)
#else
#if defined(__x86_64__) || defined(__i386__)
    // SSE supported
    #define SIMD_F32_WORD_SIZE  4
    #define SIMD_I32_WORD_SIZE  4
    #define SIMD_IS_ALIGNED(ptr)    (((size_t)(ptr)&0x0F) == 0)
#else
#ifdef __aarch64__
    // arm64 supported
    #define SIMD_F32_WORD_SIZE  4
    #define SIMD_I32_WORD_SIZE  4
    #define SIMD_IS_ALIGNED(ptr)    (1)
#else
    #define SIMD_F32_WORD_SIZE  0
    #define SIMD_I32_WORD_SIZE  0
    #define SIMD_IS_ALIGNED(ptr)    (0)
#endif  // __aarch64__
#endif  // __x86_64__ SSE
#endif  // __AVX2__
#endif  // __AVX512__

//////////////////////////////////////
// int32_t simd vector operations   //
//////////////////////////////////////
inline void my_simd_vec_i32_add(const int32_t *x, const int32_t *y, int32_t *z, const int len)
{
    int i = 0;

#if SIMD_I32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_load(&x[i]);
            simd_i_t b = my_simd_i_load(&y[i]);
            simd_i_t r = my_simd_i_add(a, b);

            my_simd_i_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_loadu(&x[i]);
            simd_i_t b = my_simd_i_loadu(&y[i]);
            simd_i_t r = my_simd_i_add(a, b);

            my_simd_i_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] + y[i];
    }
}

inline void my_simd_vec_i32_sub(const int32_t *x, const int32_t *y, int32_t *z, const int len)
{
    int i = 0;

#if SIMD_I32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_load(&x[i]);
            simd_i_t b = my_simd_i_load(&y[i]);
            simd_i_t r = my_simd_i_sub(a, b);

            my_simd_i_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_loadu(&x[i]);
            simd_i_t b = my_simd_i_loadu(&y[i]);
            simd_i_t r = my_simd_i_sub(a, b);

            my_simd_i_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] - y[i];
    }
}

inline void my_simd_vec_i32_prod(const int32_t *x, const int32_t *y, int32_t *z, const int len)
{
    int i = 0;

#if SIMD_I32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_load(&x[i]);
            simd_i_t b = my_simd_i_load(&y[i]);
            simd_i_t r = my_simd_i_mul(a, b);

            my_simd_i_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_I32_WORD_SIZE + 1; i += SIMD_I32_WORD_SIZE)
        {
            simd_i_t a = my_simd_i_loadu(&x[i]);
            simd_i_t b = my_simd_i_loadu(&y[i]);
            simd_i_t r = my_simd_i_mul(a, b);

            my_simd_i_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] * y[i];
    }
}

//////////////////////////////////////
// float32_t simd vector operations //
//////////////////////////////////////
inline void my_simd_vec_f32_add(const float *x, const float *y, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_load(&x[i]);
            simd_f_t b = my_simd_f_load(&y[i]);
            simd_f_t r = my_simd_f_add(a, b);

            my_simd_f_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_loadu(&x[i]);
            simd_f_t b = my_simd_f_loadu(&y[i]);
            simd_f_t r = my_simd_f_add(a, b);

            my_simd_f_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] + y[i];
    }
}

inline void my_simd_vec_f32_scale_add(const float *x, const float *y, const float h, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    const simd_f_t scale = my_simd_f_set1(h);

    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_load(&x[i]);
            simd_f_t b = my_simd_f_load(&y[i]);
            simd_f_t r = my_simd_f_add(a, my_simd_f_mul(b, scale));

            my_simd_f_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_loadu(&x[i]);
            simd_f_t b = my_simd_f_loadu(&y[i]);
            simd_f_t r = my_simd_f_add(a, my_simd_f_mul(b, scale));

            my_simd_f_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] + y[i] * h;
    }
}

inline void my_simd_vec_f32_sub(const float *x, const float *y, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_load(&x[i]);
            simd_f_t b = my_simd_f_load(&y[i]);
            simd_f_t r = my_simd_f_sub(a, b);

            my_simd_f_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_loadu(&x[i]);
            simd_f_t b = my_simd_f_loadu(&y[i]);
            simd_f_t r = my_simd_f_sub(a, b);

            my_simd_f_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] - y[i];
    }
}

inline void my_simd_vec_f32_prod(const float *x, const float *y, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_load(&x[i]);
            simd_f_t b = my_simd_f_load(&y[i]);
            simd_f_t r = my_simd_f_mul(a, b);

            my_simd_f_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_loadu(&x[i]);
            simd_f_t b = my_simd_f_loadu(&y[i]);
            simd_f_t r = my_simd_f_mul(a, b);

            my_simd_f_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] * y[i];
    }
}

inline void my_simd_vec_f32_div(const float *x, const float *y, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(y) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_load(&x[i]);
            simd_f_t b = my_simd_f_load(&y[i]);
            simd_f_t bcp = my_simd_f_rcp(b);
            simd_f_t r = my_simd_f_mul(a, bcp);

            my_simd_f_store(&z[i], r);
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            simd_f_t a = my_simd_f_loadu(&x[i]);
            simd_f_t b = my_simd_f_loadu(&y[i]);
            simd_f_t bcp = my_simd_f_rcp(b);
            simd_f_t r = my_simd_f_mul(a, bcp);

            my_simd_f_storeu(&z[i], r);
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = x[i] / y[i];
    }
}

inline void my_simd_vec_f32_abs(const float *x, float *z, const int len)
{
    int i = 0;

#if SIMD_F32_WORD_SIZE
    if (SIMD_IS_ALIGNED(x) && SIMD_IS_ALIGNED(z)) {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            my_simd_f_store(&z[i], my_simd_f_abs(my_simd_f_load(&x[i])));
        }
    } else {
        for (; i < len - SIMD_F32_WORD_SIZE + 1; i += SIMD_F32_WORD_SIZE)
        {
            my_simd_f_storeu(&z[i], my_simd_f_abs(my_simd_f_loadu(&x[i])));
        }
    }
#endif

    for (; i < len; i++) {
        z[i] = fabs(x[i]);
    }
}

#ifdef __cplusplus
}
#endif

#endif
