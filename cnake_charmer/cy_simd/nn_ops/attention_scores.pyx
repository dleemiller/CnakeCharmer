# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Scaled dot-product attention scores — AVX2+FMA with register hsum.

Keywords: attention, transformer, dot product, neural network, f32, simd, avx, cython
"""

from libc.math cimport sin, cos, sqrtf
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_setzero_ps()
    __m128 _mm256_castps256_ps128(__m256 a)
    __m128 _mm256_extractf128_ps(__m256 a, int imm)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    __m128 _mm_add_ss(__m128 a, __m128 b)
    __m128 _mm_movehl_ps(__m128 a, __m128 b)
    __m128 _mm_movehdup_ps(__m128 a)
    float _mm_cvtss_f32(__m128 a)


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


@cython_benchmark(syntax="cy_simd", args=(200,))
def attention_scores(int n):
    """f32 attention scores Q*K^T/sqrt(d) with AVX2+FMA dot products."""
    cdef int d = 64
    cdef float scale = 1.0 / sqrtf(<float>d)
    cdef int i, j, k

    cdef float *q_mat = <float *>malloc(n * d * sizeof(float))
    cdef float *k_mat = <float *>malloc(n * d * sizeof(float))
    if not q_mat or not k_mat:
        if q_mat: free(q_mat)
        if k_mat: free(k_mat)
        raise MemoryError()

    for i in range(n):
        for k in range(d):
            q_mat[i * d + k] = <float>sin(i * 0.1 + k * 0.01)
            k_mat[i * d + k] = <float>cos(i * 0.1 + k * 0.01)

    cdef double total = 0.0
    cdef int end16 = (d // 16) * 16
    cdef int end8 = (d // 8) * 8
    cdef __m256 vacc0, vacc1
    cdef float dot
    cdef const float *qi
    cdef const float *kj

    for i in range(n):
        qi = &q_mat[i * d]
        for j in range(n):
            kj = &k_mat[j * d]
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
            for k in range(end8, d):
                dot += qi[k] * kj[k]
            total += dot * scale

    free(q_mat)
    free(k_mat)
    return total
