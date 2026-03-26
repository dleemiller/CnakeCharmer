# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Depthwise 1D convolution on f32 tensor -- AVX2 where applicable.

Keywords: depthwise_conv, convolution, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def depthwise_conv(int n):
    """f32 depthwise 1D conv with AVX2 inner loop."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef int kernel_size = 3
    cdef int out_spatial = spatial - kernel_size + 1

    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *kernel = <float *>malloc(channels * kernel_size * sizeof(float))
    if not inp or not kernel:
        raise MemoryError()

    cdef int c, s, k, inp_offset
    cdef double total = 0.0
    cdef float val
    cdef __m256 vk, v0, v1, v2, vacc
    cdef int end8
    cdef float tmp[8]

    # Generate input
    for c in range(n):
        inp[c] = sin(c * 0.01) * 10.0

    # Generate kernel
    for c in range(channels):
        for k in range(kernel_size):
            kernel[c * kernel_size + k] = sin((c * kernel_size + k) * 0.5) * 0.5

    # Depthwise conv with AVX2 for kernel_size=3
    for c in range(channels):
        inp_offset = c * spatial
        end8 = (out_spatial // 8) * 8

        # AVX2: process 8 output elements at a time
        for s in range(0, end8, 8):
            vacc = _mm256_setzero_ps()
            for k in range(kernel_size):
                vk = _mm256_set1_ps(kernel[c * kernel_size + k])
                v0 = _mm256_loadu_ps(&inp[inp_offset + s + k])
                vacc = _mm256_add_ps(vacc, _mm256_mul_ps(v0, vk))
            _mm256_storeu_ps(tmp, vacc)
            for k in range(8):
                total += tmp[k]

        # Scalar remainder
        for s in range(end8, out_spatial):
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kernel[c * kernel_size + k]
            total += val

    free(inp)
    free(kernel)
    return total
