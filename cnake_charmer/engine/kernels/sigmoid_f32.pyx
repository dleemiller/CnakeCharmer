# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigmoid f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. sigmoid(x) = 1 / (1 + exp(-x)).
"""

from libc.math cimport expf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_div_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void sigmoid_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar sigmoid: out[i] = 1 / (1 + exp(-inp[i]))."""
    cdef int i
    for i in range(n):
        out[i] = 1.0 / (1.0 + expf(-inp[i]))


cdef void sigmoid_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """AVX2 sigmoid: loads 8 at a time, scalar exp per element, stores 8.

    No SIMD exp intrinsic available, so we batch load/store but compute
    exp element-wise via a temporary buffer.
    """
    cdef int i, j
    cdef int end8 = (n // 8) * 8
    cdef float tmp[8]

    for i in range(0, end8, 8):
        for j in range(8):
            tmp[j] = 1.0 / (1.0 + expf(-inp[i + j]))
        _mm256_storeu_ps(&out[i], _mm256_loadu_ps(tmp))

    # Scalar remainder
    for i in range(end8, n):
        out[i] = 1.0 / (1.0 + expf(-inp[i]))
