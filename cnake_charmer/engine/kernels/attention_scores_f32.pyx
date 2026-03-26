# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Attention scores f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Q*K^T / sqrt(d_model).
Q and K are (seq_len x d_model) row-major. scores is (seq_len x seq_len).
"""

from libc.math cimport sqrtf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_setzero_ps() noexcept
    # Horizontal sum intrinsics
    __m128 _mm256_castps256_ps128(__m256 a) noexcept
    __m128 _mm256_extractf128_ps(__m256 a, int imm) noexcept
    __m128 _mm_add_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_add_ss(__m128 a, __m128 b) noexcept
    __m128 _mm_movehl_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_movehdup_ps(__m128 a) noexcept
    float _mm_cvtss_f32(__m128 a) noexcept


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    """Horizontal sum — XNNPACK pattern, register-only."""
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


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
    """AVX2 attention scores: 2-accumulator FMA dot + register hsum."""
    cdef int i, j, k
    cdef int end16 = (d_model // 16) * 16
    cdef int end8 = (d_model // 8) * 8
    cdef float scale = 1.0 / sqrtf(<float>d_model)
    cdef __m256 vacc0, vacc1
    cdef float dot
    cdef const float *qi
    cdef const float *kj

    for i in range(seq_len):
        qi = &Q[i * d_model]
        for j in range(seq_len):
            kj = &K[j * d_model]

            # 2-accumulator FMA dot product
            vacc0 = _mm256_setzero_ps()
            vacc1 = _mm256_setzero_ps()
            for k in range(0, end16, 16):
                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&qi[k]),
                                        _mm256_loadu_ps(&kj[k]), vacc0)
                vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&qi[k + 8]),
                                        _mm256_loadu_ps(&kj[k + 8]), vacc1)
            for k in range(end16, end8, 8):
                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&qi[k]),
                                        _mm256_loadu_ps(&kj[k]), vacc0)
            vacc0 = _mm256_add_ps(vacc0, vacc1)
            dot = _hsum_avx(vacc0)

            # Scalar remainder
            for k in range(end8, d_model):
                dot += qi[k] * kj[k]

            scores[i * seq_len + j] = dot * scale
