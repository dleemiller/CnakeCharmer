# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Embedding lookup on f32 tensor -- AVX2 accumulation per row.

Keywords: embedding, lookup, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(500000,))
def embedding_lookup(int n):
    """f32 embedding lookup with AVX2 row accumulation."""
    cdef int vocab_size = 1000
    cdef int dim = 64
    cdef float *table = <float *>malloc(vocab_size * dim * sizeof(float))
    if not table:
        raise MemoryError()

    cdef int v, d, i, idx
    cdef int end8 = (dim // 8) * 8
    cdef double total = 0.0
    cdef float *row
    cdef __m256 acc = _mm256_setzero_ps()
    cdef float tmp[8]

    # Build embedding table
    for v in range(vocab_size):
        for d in range(dim):
            table[v * dim + d] = sin((v * dim + d) * 0.01) * 0.1

    # Lookup and sum with AVX
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

    free(table)
    return total
