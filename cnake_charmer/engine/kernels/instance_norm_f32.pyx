# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Instance normalization f32 kernels -- scalar and AVX2 implementations.

Normalize each channel independently: (x - mean) / sqrt(var + eps).
Pure compute, no allocation.
"""

from libc.math cimport sqrt

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void instance_norm_f32(const float *inp, float *out, int channels, int spatial, float eps) noexcept nogil:
    """Scalar instance normalization."""
    cdef int c, s, offset
    cdef double mean, var, diff, inv_std

    for c in range(channels):
        offset = c * spatial

        # Mean
        mean = 0.0
        for s in range(spatial):
            mean += inp[offset + s]
        mean /= spatial

        # Variance
        var = 0.0
        for s in range(spatial):
            diff = inp[offset + s] - mean
            var += diff * diff
        var /= spatial

        # Normalize
        inv_std = 1.0 / sqrt(var + eps)
        for s in range(spatial):
            out[offset + s] = <float>((inp[offset + s] - mean) * inv_std)


cdef void instance_norm_f32_avx(const float *inp, float *out, int channels, int spatial, float eps) noexcept nogil:
    """AVX2 instance normalization."""
    cdef int c, s, offset
    cdef double mean, var, diff, inv_std
    cdef int end8
    cdef __m256 vmean, vinv_std, v0
    cdef __m256 acc
    cdef float tmp[8]

    for c in range(channels):
        offset = c * spatial
        end8 = (spatial // 8) * 8

        # Mean with AVX
        acc = _mm256_setzero_ps()
        mean = 0.0
        for s in range(0, end8, 8):
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(&inp[offset + s]))
        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            mean += tmp[s]
        for s in range(end8, spatial):
            mean += inp[offset + s]
        mean /= spatial

        # Variance
        vmean = _mm256_set1_ps(<float>mean)
        acc = _mm256_setzero_ps()
        var = 0.0
        for s in range(0, end8, 8):
            v0 = _mm256_sub_ps(_mm256_loadu_ps(&inp[offset + s]), vmean)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v0, v0))
        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            var += tmp[s]
        for s in range(end8, spatial):
            diff = inp[offset + s] - mean
            var += diff * diff
        var /= spatial

        # Normalize
        inv_std = 1.0 / sqrt(var + eps)
        vinv_std = _mm256_set1_ps(<float>inv_std)
        for s in range(0, end8, 8):
            v0 = _mm256_sub_ps(_mm256_loadu_ps(&inp[offset + s]), vmean)
            _mm256_storeu_ps(&out[offset + s], _mm256_mul_ps(v0, vinv_std))
        for s in range(end8, spatial):
            out[offset + s] = <float>((inp[offset + s] - mean) * inv_std)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total
