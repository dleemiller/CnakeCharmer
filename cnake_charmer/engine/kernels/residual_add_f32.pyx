# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fused residual add + ReLU f32 kernels -- scalar and AVX2 implementations.

output = relu(input + residual) = max(0, input + residual).
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void residual_add_f32(const float *inp, const float *residual, float *out, int n) noexcept nogil:
    """Scalar fused residual add + ReLU."""
    cdef int i
    cdef float val
    for i in range(n):
        val = inp[i] + residual[i]
        out[i] = val if val > 0.0 else 0.0


cdef void residual_add_f32_avx(const float *inp, const float *residual, float *out, int n) noexcept nogil:
    """AVX2 fused residual add + ReLU: _mm256_add_ps then _mm256_max_ps."""
    cdef __m256 vzero = _mm256_setzero_ps()
    cdef __m256 v0, v1
    cdef int i
    cdef int end8 = (n // 8) * 8

    for i in range(0, end8, 8):
        v0 = _mm256_loadu_ps(&inp[i])
        v1 = _mm256_loadu_ps(&residual[i])
        v0 = _mm256_add_ps(v0, v1)
        v0 = _mm256_max_ps(vzero, v0)
        _mm256_storeu_ps(&out[i], v0)

    # Scalar remainder
    cdef float val
    for i in range(end8, n):
        val = inp[i] + residual[i]
        out[i] = val if val > 0.0 else 0.0


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
