# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Global average pool f32 kernels -- scalar and AVX2 implementations.

Mean per channel over spatial dimension.
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void global_avg_pool_f32(const float *inp, float *out, int channels, int spatial) noexcept nogil:
    """Scalar global average pool: mean per channel."""
    cdef int c, s, offset
    cdef double channel_sum
    cdef float inv_spatial = 1.0 / spatial

    for c in range(channels):
        offset = c * spatial
        channel_sum = 0.0
        for s in range(spatial):
            channel_sum += inp[offset + s]
        out[c] = <float>(channel_sum * inv_spatial)


cdef void global_avg_pool_f32_avx(const float *inp, float *out, int channels, int spatial) noexcept nogil:
    """AVX2 global average pool."""
    cdef int c, s, offset
    cdef int end8 = (spatial // 8) * 8
    cdef __m256 acc
    cdef float tmp[8]
    cdef double channel_sum
    cdef float inv_spatial = 1.0 / spatial

    for c in range(channels):
        offset = c * spatial
        acc = _mm256_setzero_ps()
        channel_sum = 0.0

        for s in range(0, end8, 8):
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(&inp[offset + s]))

        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            channel_sum += tmp[s]
        for s in range(end8, spatial):
            channel_sum += inp[offset + s]

        out[c] = <float>(channel_sum * inv_spatial)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total
