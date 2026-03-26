# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Softmax f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Numerically stable: find max, subtract, exp, sum, divide.
"""

from libc.math cimport expf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_div_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void softmax_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar numerically stable softmax."""
    cdef int i
    cdef float mx, s, val

    # Pass 1: find max
    mx = inp[0]
    for i in range(1, n):
        if inp[i] > mx:
            mx = inp[i]

    # Pass 2: exp(x - max) and sum
    s = 0.0
    for i in range(n):
        val = expf(inp[i] - mx)
        out[i] = val
        s += val

    # Pass 3: normalize
    cdef float inv_s = 1.0 / s
    for i in range(n):
        out[i] *= inv_s


cdef void softmax_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """AVX2 softmax: vectorized max-find, scalar exp, vectorized normalize.

    No SIMD exp intrinsic, so exp is computed element-wise.
    Max-find and final division are fully vectorized.
    """
    cdef int i, j
    cdef int end8 = (n // 8) * 8
    cdef __m256 vmax, v0, vsum, vinv_s
    cdef float tmp[8]
    cdef float mx, s

    if n == 0:
        return

    # Pass 1: vectorized max
    if end8 > 0:
        vmax = _mm256_loadu_ps(&inp[0])
        for i in range(8, end8, 8):
            v0 = _mm256_loadu_ps(&inp[i])
            vmax = _mm256_max_ps(vmax, v0)
        _mm256_storeu_ps(tmp, vmax)
        mx = tmp[0]
        for j in range(1, 8):
            if tmp[j] > mx:
                mx = tmp[j]
    else:
        mx = inp[0]
    for i in range(end8, n):
        if inp[i] > mx:
            mx = inp[i]

    # Pass 2: exp(x - max) — scalar exp, store results
    s = 0.0
    for i in range(n):
        out[i] = expf(inp[i] - mx)
        s += out[i]

    # Pass 3: vectorized normalize
    cdef float inv_s = 1.0 / s
    vinv_s = _mm256_set1_ps(inv_s)
    for i in range(0, end8, 8):
        v0 = _mm256_loadu_ps(&out[i])
        v0 = _mm256_mul_ps(v0, vinv_s)
        _mm256_storeu_ps(&out[i], v0)
    for i in range(end8, n):
        out[i] *= inv_s
