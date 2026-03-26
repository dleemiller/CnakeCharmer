# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fused residual add + ReLU on f32 tensor -- AVX2 vectorized.

output = relu(input + residual): _mm256_add_ps then _mm256_max_ps.

Keywords: residual, add, relu, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_max_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def residual_add(int n):
    """f32 fused residual add + ReLU with AVX2."""
    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *residual = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not inp or not residual or not out:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float val

    # Generate input and residual
    for i in range(n):
        inp[i] = sin(i * 0.01) * 10.0
        residual[i] = cos(i * 0.01) * 10.0

    # Fused residual add + ReLU with AVX2
    cdef __m256 vzero = _mm256_setzero_ps()
    cdef __m256 v0, v1
    cdef int end8 = (n // 8) * 8

    for i in range(0, end8, 8):
        v0 = _mm256_loadu_ps(&inp[i])
        v1 = _mm256_loadu_ps(&residual[i])
        v0 = _mm256_add_ps(v0, v1)
        v0 = _mm256_max_ps(vzero, v0)
        _mm256_storeu_ps(&out[i], v0)

    # Scalar remainder
    for i in range(end8, n):
        val = inp[i] + residual[i]
        if val < 0.0:
            val = 0.0
        out[i] = val

    # AVX reduction
    cdef __m256 acc = _mm256_setzero_ps()
    cdef float tmp[8]

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&out[i]))
    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, n):
        total += out[i]

    free(inp)
    free(residual)
    free(out)
    return total
