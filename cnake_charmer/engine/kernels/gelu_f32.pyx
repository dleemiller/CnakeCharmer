# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GELU f32 kernels -- scalar and AVX2 implementations.

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
Pure compute, no allocation.
"""

from libc.math cimport tanh

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept

cdef float SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
cdef float GELU_COEFF = 0.044715


cdef void gelu_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar GELU: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))."""
    cdef int i
    cdef float x, inner
    for i in range(n):
        x = inp[i]
        inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
        out[i] = 0.5 * x * (1.0 + tanh(inner))


cdef void gelu_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """AVX2 GELU -- scalar fallback (tanh has no AVX intrinsic)."""
    # GELU requires tanh which has no direct AVX intrinsic, use scalar
    gelu_f32(inp, out, n)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total


cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil:
    """AVX2 sum reduction: 8-wide accumulate."""
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
