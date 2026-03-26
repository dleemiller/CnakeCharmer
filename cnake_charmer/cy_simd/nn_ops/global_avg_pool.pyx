# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Global average pooling on f32 tensor -- AVX2 vectorized.

channels=64, spatial=n/64. Return sum of channel means.

Keywords: global_avg_pool, pooling, neural network, tensor, f32, simd, avx, cython
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


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def global_avg_pool(int n):
    """f32 global average pool with AVX2."""
    cdef int channels = 64
    cdef int spatial = n // channels

    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int c, s, offset
    cdef double channel_sum
    cdef double total = 0.0
    cdef int end8 = (spatial // 8) * 8
    cdef __m256 acc
    cdef float tmp[8]

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    # Global average pool with AVX per channel
    for c in range(channels):
        offset = c * spatial
        acc = _mm256_setzero_ps()
        channel_sum = 0.0

        for s in range(0, end8, 8):
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[offset + s]))

        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            channel_sum += tmp[s]
        for s in range(end8, spatial):
            channel_sum += data[offset + s]

        total += channel_sum / spatial

    free(data)
    return total
