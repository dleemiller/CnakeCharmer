# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU on f32 tensor — XNNPACK-style AVX vclamp pattern.

Processes 16 floats per outer iteration (2x __m256), matching the
XNNPACK f32-vclamp-avx-u16 microkernel pattern:
  load 16 → max(zero) 16 → store 16

Keywords: relu, activation, neural network, tensor, f32, simd, avx, xnnpack, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    # f32 AVX types and ops (matches XNNPACK)
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_max_ps(__m256 a, __m256 b)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def relu(int n):
    """f32 ReLU with AVX — 16 floats per iteration, XNNPACK-style."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0

    # Allocate tensor
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # ReLU in-place — XNNPACK pattern: batch-16 with 2x __m256
    cdef __m256 vzero = _mm256_setzero_ps()
    cdef __m256 v0, v1
    cdef int batch16_end = (n // 16) * 16
    cdef int batch8_end = (n // 8) * 8

    # Main loop: 16 floats per iteration
    for i in range(0, batch16_end, 16):
        v0 = _mm256_loadu_ps(&data[i])
        v1 = _mm256_loadu_ps(&data[i + 8])
        v0 = _mm256_max_ps(vzero, v0)
        v1 = _mm256_max_ps(vzero, v1)
        _mm256_storeu_ps(&data[i], v0)
        _mm256_storeu_ps(&data[i + 8], v1)

    # Remainder: 8 floats
    for i in range(batch16_end, batch8_end, 8):
        v0 = _mm256_loadu_ps(&data[i])
        v0 = _mm256_max_ps(vzero, v0)
        _mm256_storeu_ps(&data[i], v0)

    # Scalar remainder
    for i in range(batch8_end, n):
        if data[i] < 0.0:
            data[i] = 0.0

    # Reduce with AVX
    cdef __m256 acc = _mm256_setzero_ps()
    cdef float tmp[8]
    for i in range(0, batch8_end, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[i]))
    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(batch8_end, n):
        total += data[i]

    free(data)
    return total
