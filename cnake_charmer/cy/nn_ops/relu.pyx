# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU activation function with AVX2 SIMD vectorization.

Processes 8 integers at a time using 256-bit SIMD registers.
Demonstrates the XNNPACK pattern: load → compute → accumulate.

Keywords: relu, activation, neural network, simd, avx2, cython
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef int __m256i
    __m256i _mm256_loadu_si256(const __m256i *mem)
    __m256i _mm256_max_epi32(__m256i a, __m256i b)
    __m256i _mm256_setzero_si256()
    __m256i _mm256_add_epi32(__m256i a, __m256i b)
    void _mm256_storeu_si256(__m256i *mem, __m256i a)


@cython_benchmark(syntax="cy", args=(10000000,))
def relu(int n):
    """Apply ReLU using AVX2 _mm256_max_epi32 (8 ints at a time)."""
    cdef int *data = <int *>malloc(n * sizeof(int))
    if not data:
        raise MemoryError()

    cdef int i
    cdef long long total = 0

    # Generate values
    for i in range(n):
        data[i] = (i * 17 + 5) % 201 - 100

    # SIMD ReLU: max(0, x) for 8 elements at a time
    cdef __m256i zero = _mm256_setzero_si256()
    cdef __m256i vec, result
    cdef __m256i acc = _mm256_setzero_si256()
    cdef int simd_end = (n // 8) * 8
    cdef int tmp[8]

    for i in range(0, simd_end, 8):
        vec = _mm256_loadu_si256(<__m256i *>&data[i])
        result = _mm256_max_epi32(vec, zero)
        acc = _mm256_add_epi32(acc, result)

    # Extract accumulated sum from SIMD register
    _mm256_storeu_si256(<__m256i *>tmp, acc)
    for i in range(8):
        total += tmp[i]

    # Handle remainder
    for i in range(simd_end, n):
        if data[i] > 0:
            total += data[i]

    free(data)
    return int(total)
