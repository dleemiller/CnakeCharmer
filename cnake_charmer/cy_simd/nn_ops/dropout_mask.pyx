# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Deterministic dropout mask on f32 tensor -- scalar with AVX reduction.

Keywords: dropout, mask, neural network, tensor, f32, simd, avx, cython
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


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def dropout_mask(int n):
    """f32 dropout mask -- scalar mask, AVX reduction."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not data or not out:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float p = 0.1
    cdef float scale = 1.0 / (1.0 - p)
    cdef int threshold = <int>(p * 100)

    # Generate input
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # Apply dropout mask (index-dependent, scalar)
    for i in range(n):
        if (i * 7 + 3) % 100 >= threshold:
            out[i] = data[i] * scale
        else:
            out[i] = 0.0

    # AVX reduction
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int end8 = (n // 8) * 8
    cdef float tmp[8]

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&out[i]))
    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, n):
        total += out[i]

    free(data)
    free(out)
    return total
