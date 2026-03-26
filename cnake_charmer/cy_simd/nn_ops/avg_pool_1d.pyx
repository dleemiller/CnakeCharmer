# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Average pooling 1D on f32 tensor -- AVX2 reduction.

Average pooling with kernel=4, stride=4.

Keywords: avg_pool, pooling, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def avg_pool_1d(int n):
    """f32 avg pool 1D with AVX2."""
    cdef float *signal = <float *>malloc(n * sizeof(float))
    if not signal:
        raise MemoryError()

    cdef int out_len = n // 4
    cdef float *pooled = <float *>malloc(out_len * sizeof(float))
    if not pooled:
        free(signal)
        raise MemoryError()

    cdef int i, base
    cdef double total = 0.0

    # Generate signal
    for i in range(n):
        signal[i] = (i * 31 + 17) % 1000 / 10.0

    # Average pool kernel=4, stride=4
    for i in range(out_len):
        base = i * 4
        pooled[i] = (signal[base] + signal[base + 1] + signal[base + 2] + signal[base + 3]) * 0.25

    # AVX reduction on pooled
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int end8 = (out_len // 8) * 8
    cdef float tmp[8]

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&pooled[i]))
    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, out_len):
        total += pooled[i]

    free(signal)
    free(pooled)
    return total
