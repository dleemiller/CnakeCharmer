# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Elementwise add f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. output[i] = a[i] + b[i].
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept


cdef void elementwise_add_f32(const float *a, const float *b,
                              float *out, int n) noexcept nogil:
    """Scalar elementwise add."""
    cdef int i
    for i in range(n):
        out[i] = a[i] + b[i]


cdef void elementwise_add_f32_avx(const float *a, const float *b,
                                  float *out, int n) noexcept nogil:
    """AVX2 elementwise add: 16 floats per iteration, 8-wide remainder."""
    cdef __m256 va0, va1, vb0, vb1
    cdef int i
    cdef int end16 = (n // 16) * 16
    cdef int end8 = (n // 8) * 8

    # Main loop: 16 per iteration
    for i in range(0, end16, 16):
        va0 = _mm256_loadu_ps(&a[i])
        va1 = _mm256_loadu_ps(&a[i + 8])
        vb0 = _mm256_loadu_ps(&b[i])
        vb1 = _mm256_loadu_ps(&b[i + 8])
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va0, vb0))
        _mm256_storeu_ps(&out[i + 8], _mm256_add_ps(va1, vb1))

    # 8-wide remainder
    for i in range(end16, end8, 8):
        va0 = _mm256_loadu_ps(&a[i])
        vb0 = _mm256_loadu_ps(&b[i])
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va0, vb0))

    # Scalar remainder
    for i in range(end8, n):
        out[i] = a[i] + b[i]
