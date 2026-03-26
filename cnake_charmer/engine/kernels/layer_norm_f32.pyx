# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Layer norm f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Two-pass: compute mean and variance,
then normalize: out[i] = (inp[i] - mean) / sqrt(var + epsilon).
"""

from libc.math cimport sqrtf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void layer_norm_f32(const float *inp, float *out,
                         int n, float epsilon) noexcept nogil:
    """Scalar layer norm: two-pass mean+var, then normalize."""
    cdef int i
    cdef float mean = 0.0
    cdef float var = 0.0
    cdef float diff, inv_std

    # Pass 1: mean
    for i in range(n):
        mean += inp[i]
    mean /= n

    # Pass 2: variance
    for i in range(n):
        diff = inp[i] - mean
        var += diff * diff
    var /= n

    # Normalize
    inv_std = 1.0 / sqrtf(var + epsilon)
    for i in range(n):
        out[i] = (inp[i] - mean) * inv_std


cdef void layer_norm_f32_avx(const float *inp, float *out,
                             int n, float epsilon) noexcept nogil:
    """AVX2 layer norm: vectorized accumulation for mean and variance."""
    cdef int i, j
    cdef int end8 = (n // 8) * 8
    cdef __m256 vsum, v0, vmean, vdiff, vvar, vinv_std
    cdef float tmp[8]
    cdef float mean = 0.0
    cdef float var = 0.0
    cdef float inv_std

    # Pass 1: vectorized mean accumulation
    vsum = _mm256_setzero_ps()
    for i in range(0, end8, 8):
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&inp[i]))
    _mm256_storeu_ps(tmp, vsum)
    for j in range(8):
        mean += tmp[j]
    for i in range(end8, n):
        mean += inp[i]
    mean /= n

    # Pass 2: vectorized variance accumulation
    vmean = _mm256_set1_ps(mean)
    vsum = _mm256_setzero_ps()
    for i in range(0, end8, 8):
        vdiff = _mm256_sub_ps(_mm256_loadu_ps(&inp[i]), vmean)
        vsum = _mm256_fmadd_ps(vdiff, vdiff, vsum)
    _mm256_storeu_ps(tmp, vsum)
    for j in range(8):
        var += tmp[j]
    for i in range(end8, n):
        var += (inp[i] - mean) * (inp[i] - mean)
    var /= n

    # Pass 3: vectorized normalize
    inv_std = 1.0 / sqrtf(var + epsilon)
    vinv_std = _mm256_set1_ps(inv_std)
    for i in range(0, end8, 8):
        v0 = _mm256_sub_ps(_mm256_loadu_ps(&inp[i]), vmean)
        v0 = _mm256_mul_ps(v0, vinv_std)
        _mm256_storeu_ps(&out[i], v0)
    for i in range(end8, n):
        out[i] = (inp[i] - mean) * inv_std
