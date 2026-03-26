# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dropout mask f32 kernels -- scalar and AVX2 implementations.

Deterministic dropout: mask[i] = 1 if (i*7+3)%100 >= p*100 else 0.
output[i] = input[i] * mask[i] / (1 - p).
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void dropout_mask_f32(const float *inp, float *out, int n, float p) noexcept nogil:
    """Scalar dropout mask application."""
    cdef int i
    cdef int threshold = <int>(p * 100)
    cdef float scale = 1.0 / (1.0 - p)
    for i in range(n):
        if (i * 7 + 3) % 100 >= threshold:
            out[i] = inp[i] * scale
        else:
            out[i] = 0.0


cdef void dropout_mask_f32_avx(const float *inp, float *out, int n, float p) noexcept nogil:
    """AVX2 dropout -- scalar fallback (mask is index-dependent)."""
    dropout_mask_f32(inp, out, n, p)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total
