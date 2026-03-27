# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Depthwise 1D convolution — AVX2+FMA broadcast kernel pattern.

Keywords: depthwise_conv, convolution, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_broadcast_ss(const float *mem)
    __m256 _mm256_setzero_ps()
    __m128 _mm256_castps256_ps128(__m256 a)
    __m128 _mm256_extractf128_ps(__m256 a, int imm)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    __m128 _mm_add_ss(__m128 a, __m128 b)
    __m128 _mm_movehl_ps(__m128 a, __m128 b)
    __m128 _mm_movehdup_ps(__m128 a)
    float _mm_cvtss_f32(__m128 a)


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def depthwise_conv(int n):
    """f32 depthwise 1D conv with AVX2+FMA — broadcast kernel, 8 outputs at a time."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef int kernel_size = 3
    cdef int out_spatial = spatial - kernel_size + 1

    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *kern = <float *>malloc(channels * kernel_size * sizeof(float))
    cdef float *out = <float *>malloc(channels * out_spatial * sizeof(float))
    if not inp or not kern or not out:
        raise MemoryError()

    cdef int c, s, k, inp_offset, kern_offset, out_offset
    cdef double total = 0.0
    cdef float val
    cdef __m256 vacc, vk

    # Cheap input gen (no sin)
    for c in range(n):
        inp[c] = <float>sin(c * 0.01) * 10.0

    # Generate kernel
    for c in range(channels):
        for k in range(kernel_size):
            kern[c * kernel_size + k] = sin((c * kernel_size + k) * 0.5) * 0.5

    # Depthwise conv: broadcast+FMA per kernel tap
    for c in range(channels):
        inp_offset = c * spatial
        kern_offset = c * kernel_size
        out_offset = c * out_spatial

        s = 0
        while s + 8 <= out_spatial:
            vacc = _mm256_setzero_ps()
            for k in range(kernel_size):
                vk = _mm256_broadcast_ss(&kern[kern_offset + k])
                vacc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(&inp[inp_offset + s + k]), vacc)
            _mm256_storeu_ps(&out[out_offset + s], vacc)
            s += 8

        while s < out_spatial:
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kern[kern_offset + k]
            out[out_offset + s] = val
            s += 1

    # Reduce output with AVX
    cdef int total_out = channels * out_spatial
    cdef int rend8 = (total_out // 8) * 8
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    cdef int rend16 = (total_out // 16) * 16
    for s in range(0, rend16, 16):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[s]))
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&out[s + 8]))
    for s in range(rend16, rend8, 8):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[s]))
    acc0 = _mm256_add_ps(acc0, acc1)
    total = <double>_hsum_avx(acc0)
    for s in range(rend8, total_out):
        total += out[s]

    free(inp); free(kern); free(out)
    return total
