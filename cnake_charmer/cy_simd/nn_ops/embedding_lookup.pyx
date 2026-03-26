# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Embedding lookup — AVX2 with 4-accumulator row accumulation.

Keywords: embedding, lookup, neural network, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
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


@cython_benchmark(syntax="cy_simd", args=(500000,))
def embedding_lookup(int n):
    """f32 embedding lookup with AVX2 — 4-accumulator rsum pattern."""
    cdef int vocab_size = 1000
    cdef int dim = 64
    cdef float *table = <float *>malloc(vocab_size * dim * sizeof(float))
    if not table:
        raise MemoryError()

    cdef int v, d, i, idx
    cdef int end32 = (dim // 32) * 32
    cdef int end8 = (dim // 8) * 8
    cdef float *row

    # Build embedding table
    for v in range(vocab_size):
        for d in range(dim):
            table[v * dim + d] = sin((v * dim + d) * 0.01) * 0.1

    # Lookup and sum with 4-accumulator AVX
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    cdef __m256 acc2 = _mm256_setzero_ps()
    cdef __m256 acc3 = _mm256_setzero_ps()
    cdef double tail_total = 0.0

    for i in range(n):
        idx = (i * 7 + 3) % vocab_size
        row = &table[idx * dim]
        for d in range(0, end32, 32):
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&row[d]))
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&row[d + 8]))
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(&row[d + 16]))
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(&row[d + 24]))
        for d in range(end32, end8, 8):
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&row[d]))
        for d in range(end8, dim):
            tail_total += row[d]

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))
    cdef double total = <double>_hsum_avx(acc0) + tail_total

    free(table)
    return total
