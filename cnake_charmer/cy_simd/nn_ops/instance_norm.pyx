# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Instance normalization on f32 tensor -- AVX2 vectorized.

Keywords: instance_norm, normalization, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, sqrt
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def instance_norm(int n):
    """f32 instance normalization with AVX2."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef float eps = 1e-5

    cdef float *data = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not data or not out:
        raise MemoryError()

    cdef int c, s, offset
    cdef double mean, var, diff, inv_std
    cdef double total = 0.0
    cdef int end8
    cdef __m256 acc, vmean, vinv_std, v0
    cdef float tmp[8]

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    for c in range(channels):
        offset = c * spatial
        end8 = (spatial // 8) * 8

        # Mean with AVX
        acc = _mm256_setzero_ps()
        mean = 0.0
        for s in range(0, end8, 8):
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[offset + s]))
        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            mean += tmp[s]
        for s in range(end8, spatial):
            mean += data[offset + s]
        mean /= spatial

        # Variance with AVX
        vmean = _mm256_set1_ps(<float>mean)
        acc = _mm256_setzero_ps()
        var = 0.0
        for s in range(0, end8, 8):
            v0 = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s]), vmean)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v0, v0))
        _mm256_storeu_ps(tmp, acc)
        for s in range(8):
            var += tmp[s]
        for s in range(end8, spatial):
            diff = data[offset + s] - mean
            var += diff * diff
        var /= spatial

        # Normalize with AVX
        inv_std = 1.0 / sqrt(var + eps)
        vinv_std = _mm256_set1_ps(<float>inv_std)
        for s in range(0, end8, 8):
            v0 = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s]), vmean)
            _mm256_storeu_ps(&out[offset + s], _mm256_mul_ps(v0, vinv_std))
        for s in range(end8, spatial):
            out[offset + s] = <float>((data[offset + s] - mean) * inv_std)

    # Reduce with AVX
    acc = _mm256_setzero_ps()
    end8 = (n // 8) * 8
    for s in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&out[s]))
    _mm256_storeu_ps(tmp, acc)
    for s in range(8):
        total += tmp[s]
    for s in range(end8, n):
        total += out[s]

    free(data)
    free(out)
    return total
