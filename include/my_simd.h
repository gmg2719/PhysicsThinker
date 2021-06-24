#ifndef _MY_SIMD_H_
#define _MY_SIMD_H_     1

#if defined(__x86_64__) || defined(__i386__)
    include <immintrin.h>
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

//////////////////////////////////////
// int32_t simd basic operations    //       
//////////////////////////////////////
static inline simd_i_t my_simd_i_load(int32_t *x)
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

static inline simd_i_t my_simd_i_loadu(int32_t *x)
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

#endif
