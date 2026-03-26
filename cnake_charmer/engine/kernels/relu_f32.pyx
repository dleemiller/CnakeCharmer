# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Can be cimported by benchmark wrappers
or a future inference engine.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void relu_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar ReLU: out[i] = max(0, inp[i]). Supports in-place (inp == out)."""
    cdef int i
    for i in range(n):
        out[i] = inp[i] if inp[i] > 0.0 else 0.0


cdef void relu_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """AVX2 ReLU: XNNPACK vclamp pattern, 16 floats per iteration."""
    cdef __m256 vzero = _mm256_setzero_ps()
    cdef __m256 v0, v1
    cdef int i
    cdef int end16 = (n // 16) * 16
    cdef int end8 = (n // 8) * 8

    # Main loop: 16 per iteration
    for i in range(0, end16, 16):
        v0 = _mm256_loadu_ps(&inp[i])
        v1 = _mm256_loadu_ps(&inp[i + 8])
        v0 = _mm256_max_ps(vzero, v0)
        v1 = _mm256_max_ps(vzero, v1)
        _mm256_storeu_ps(&out[i], v0)
        _mm256_storeu_ps(&out[i + 8], v1)

    # 8-wide remainder
    for i in range(end16, end8, 8):
        v0 = _mm256_loadu_ps(&inp[i])
        v0 = _mm256_max_ps(vzero, v0)
        _mm256_storeu_ps(&out[i], v0)

    # Scalar remainder
    for i in range(end8, n):
        out[i] = inp[i] if inp[i] > 0.0 else 0.0


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
