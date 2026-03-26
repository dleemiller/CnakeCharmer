# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Numerically stable softmax — AVX2 for max-reduction and normalization.

Keywords: softmax, stable, neural network, activation, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport expf
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_max_ps(__m256 a, __m256 b)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()
    __m128 _mm256_castps256_ps128(__m256 a)
    __m128 _mm256_extractf128_ps(__m256 a, int imm)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    __m128 _mm_add_ss(__m128 a, __m128 b)
    __m128 _mm_max_ps(__m128 a, __m128 b)
    __m128 _mm_max_ss(__m128 a, __m128 b)
    __m128 _mm_movehl_ps(__m128 a, __m128 b)
    __m128 _mm_movehdup_ps(__m128 a)
    float _mm_cvtss_f32(__m128 a)


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


cdef inline float _hmax_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 m = _mm_max_ps(lo, hi)
    m = _mm_max_ps(m, _mm_movehl_ps(m, m))
    m = _mm_max_ss(m, _mm_movehdup_ps(m))
    return _mm_cvtss_f32(m)


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def softmax_stable(int n):
    """f32 stable softmax on groups of 1024 with AVX2 max-reduce."""
    cdef int group_size = 100
    cdef int num_groups = n // group_size
    cdef int end8 = (group_size // 8) * 8

    cdef float *vals = <float *>malloc(group_size * sizeof(float))
    cdef float *exp_vals = <float *>malloc(group_size * sizeof(float))
    if not vals or not exp_vals:
        raise MemoryError()

    cdef int g, i, offset
    cdef double total = 0.0
    cdef float max_val, exp_sum, inv_sum
    cdef __m256 vmax, vacc, vinv

    for g in range(num_groups):
        offset = g * group_size
        for i in range(group_size):
            vals[i] = <float>(((offset + i) * 17 + 5) % 1000) / 100.0

        # AVX max reduction
        vmax = _mm256_loadu_ps(&vals[0])
        for i in range(8, end8, 8):
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(&vals[i]))
        max_val = _hmax_avx(vmax)
        for i in range(end8, group_size):
            if vals[i] > max_val:
                max_val = vals[i]

        # Exp + sum (scalar — no AVX exp)
        exp_sum = 0.0
        for i in range(group_size):
            exp_vals[i] = expf(vals[i] - max_val)
            exp_sum += exp_vals[i]

        # Normalize with AVX
        inv_sum = 1.0 / exp_sum
        vinv = _mm256_set1_ps(inv_sum)
        vacc = _mm256_setzero_ps()
        for i in range(0, end8, 8):
            _mm256_storeu_ps(&exp_vals[i], _mm256_mul_ps(_mm256_loadu_ps(&exp_vals[i]), vinv))
            vacc = _mm256_add_ps(vacc, _mm256_loadu_ps(&exp_vals[i]))
        total += <double>_hsum_avx(vacc)
        for i in range(end8, group_size):
            total += exp_vals[i] * inv_sum

    free(vals)
    free(exp_vals)
    return total
