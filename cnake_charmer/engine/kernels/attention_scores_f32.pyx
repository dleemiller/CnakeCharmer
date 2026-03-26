# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Attention scores f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Q*K^T / sqrt(d_model).
Q and K are (seq_len x d_model) row-major. scores is (seq_len x seq_len).
"""

from libc.math cimport sqrtf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void attention_scores_f32(const float *Q, const float *K, float *scores,
                               int seq_len, int d_model) noexcept nogil:
    """Scalar attention scores: scores[i][j] = dot(Q[i], K[j]) / sqrt(d)."""
    cdef int i, j, k
    cdef float dot, scale
    scale = 1.0 / sqrtf(<float>d_model)

    for i in range(seq_len):
        for j in range(seq_len):
            dot = 0.0
            for k in range(d_model):
                dot += Q[i * d_model + k] * K[j * d_model + k]
            scores[i * seq_len + j] = dot * scale


cdef void attention_scores_f32_avx(const float *Q, const float *K, float *scores,
                                   int seq_len, int d_model) noexcept nogil:
    """AVX2 attention scores: FMA dot product over d_model dimension."""
    cdef int i, j, k
    cdef int end8 = (d_model // 8) * 8
    cdef float scale = 1.0 / sqrtf(<float>d_model)
    cdef __m256 vacc, vq, vk
    cdef float tmp[8]
    cdef float dot

    for i in range(seq_len):
        for j in range(seq_len):
            # Vectorized dot product
            vacc = _mm256_setzero_ps()
            for k in range(0, end8, 8):
                vq = _mm256_loadu_ps(&Q[i * d_model + k])
                vk = _mm256_loadu_ps(&K[j * d_model + k])
                vacc = _mm256_fmadd_ps(vq, vk, vacc)

            # Horizontal sum
            _mm256_storeu_ps(tmp, vacc)
            dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

            # Scalar remainder
            for k in range(end8, d_model):
                dot += Q[i * d_model + k] * K[j * d_model + k]

            scores[i * seq_len + j] = dot * scale
