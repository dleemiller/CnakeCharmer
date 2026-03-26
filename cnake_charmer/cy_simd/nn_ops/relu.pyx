# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU on a C array tensor with AVX2 SIMD vectorization.

Same 3-step pattern as basic Cython (allocate → ReLU in-place → reduce)
but both the ReLU and reduce steps use AVX2 to process 8 ints per cycle.

In a real ResNet forward pass, this tensor would come from a previous
conv2d layer and flow into the next. The SIMD advantage compounds
across many layers.

Keywords: relu, activation, neural network, tensor, simd, avx2, cython
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef int __m256i
    __m256i _mm256_loadu_si256(const __m256i *mem)
    void _mm256_storeu_si256(__m256i *mem, __m256i a)
    __m256i _mm256_max_epi32(__m256i a, __m256i b)
    __m256i _mm256_add_epi32(__m256i a, __m256i b)
    __m256i _mm256_setzero_si256()


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def relu(int n):
    """Allocate C array tensor, apply ReLU in-place with AVX2, return sum."""
    cdef int *data = <int *>malloc(n * sizeof(int))
    if not data:
        raise MemoryError()

    cdef int i
    cdef long long total = 0
    cdef int simd_end = (n // 8) * 8

    # Step 1: Allocate tensor (same as basic Cython)
    for i in range(n):
        data[i] = (i * 17 + 5) % 201 - 100

    # Step 2: ReLU in-place with AVX2 — 8 elements per instruction
    cdef __m256i zero = _mm256_setzero_si256()
    cdef __m256i vec
    for i in range(0, simd_end, 8):
        vec = _mm256_loadu_si256(<__m256i *>&data[i])
        vec = _mm256_max_epi32(vec, zero)
        _mm256_storeu_si256(<__m256i *>&data[i], vec)
    for i in range(simd_end, n):
        if data[i] < 0:
            data[i] = 0

    # Step 3: Reduce with AVX2 — accumulate 8 at a time
    cdef __m256i acc = _mm256_setzero_si256()
    cdef int tmp[8]
    for i in range(0, simd_end, 8):
        vec = _mm256_loadu_si256(<__m256i *>&data[i])
        acc = _mm256_add_epi32(acc, vec)
    _mm256_storeu_si256(<__m256i *>tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(simd_end, n):
        total += data[i]

    free(data)
    return int(total)
