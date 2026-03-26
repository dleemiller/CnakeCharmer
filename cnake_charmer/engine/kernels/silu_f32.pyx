# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SiLU/Swish f32 kernels -- scalar and AVX2 implementations.

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
Pure compute, no allocation.
"""

from libc.math cimport exp

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void silu_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar SiLU: out[i] = x / (1 + exp(-x))."""
    cdef int i
    cdef float x
    for i in range(n):
        x = inp[i]
        out[i] = x / (1.0 + exp(-<double>x))


cdef void silu_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """AVX2 SiLU -- scalar fallback (exp has no AVX intrinsic)."""
    silu_f32(inp, out, n)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total


cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil:
    """AVX2 sum reduction."""
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int i
    cdef int end8 = (n // 8) * 8
    cdef float tmp[8]
    cdef double total = 0.0

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[i]))

    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, n):
        total += data[i]

    return total
