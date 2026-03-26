# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch norm f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Inference mode:
output[i] = gamma * (input[i] - mean) * inv_std + beta
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept


cdef void batch_norm_f32(const float *inp, float *out, int n,
                         float mean, float inv_std,
                         float gamma, float beta) noexcept nogil:
    """Scalar batch norm inference."""
    cdef int i
    cdef float scale = gamma * inv_std
    for i in range(n):
        out[i] = scale * (inp[i] - mean) + beta


cdef void batch_norm_f32_avx(const float *inp, float *out, int n,
                             float mean, float inv_std,
                             float gamma, float beta) noexcept nogil:
    """AVX2 batch norm: broadcast params, FMA, store 8 at a time."""
    cdef float scale = gamma * inv_std
    cdef __m256 vmean = _mm256_set1_ps(mean)
    cdef __m256 vscale = _mm256_set1_ps(scale)
    cdef __m256 vbeta = _mm256_set1_ps(beta)
    cdef __m256 v0
    cdef int i
    cdef int end8 = (n // 8) * 8

    for i in range(0, end8, 8):
        v0 = _mm256_loadu_ps(&inp[i])
        v0 = _mm256_sub_ps(v0, vmean)
        v0 = _mm256_fmadd_ps(vscale, v0, vbeta)
        _mm256_storeu_ps(&out[i], v0)

    # Scalar remainder
    for i in range(end8, n):
        out[i] = scale * (inp[i] - mean) + beta
