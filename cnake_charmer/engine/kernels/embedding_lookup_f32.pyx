# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Embedding lookup f32 kernels -- scalar and AVX2 implementations.

Lookup n embeddings of given dim from a table.
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef double embedding_lookup_f32(const float *table, int vocab_size, int dim, int n) noexcept nogil:
    """Scalar embedding lookup: sum all looked-up values."""
    cdef double total = 0.0
    cdef int i, d, idx
    cdef const float *row
    for i in range(n):
        idx = (i * 7 + 3) % vocab_size
        row = &table[idx * dim]
        for d in range(dim):
            total += row[d]
    return total


cdef double embedding_lookup_f32_avx(const float *table, int vocab_size, int dim, int n) noexcept nogil:
    """AVX2 embedding lookup: 8-wide accumulate per row."""
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int i, d, idx
    cdef int end8 = (dim // 8) * 8
    cdef const float *row
    cdef float tmp[8]
    cdef double total = 0.0

    for i in range(n):
        idx = (i * 7 + 3) % vocab_size
        row = &table[idx * dim]
        for d in range(0, end8, 8):
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(&row[d]))
        for d in range(end8, dim):
            total += row[d]

    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]

    return total
