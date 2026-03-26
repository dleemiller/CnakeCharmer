# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch normalization — AVX2+FMA vectorized.

Keywords: batch norm, normalization, neural network, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_set1_ps(float a)
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


@cython_benchmark(syntax="cy_simd", args=(2000000,))
def batch_norm(int n):
    """f32 batch norm with AVX2+FMA: gamma*(x-mean)*inv_std + beta."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not data or not out:
        raise MemoryError()

    cdef int i
    cdef int end32 = (n // 32) * 32
    cdef int end8 = (n // 8) * 8

    # Fill input
    for i in range(n):
        data[i] = ((i * 17 + 5) % 1000) / 10.0

    # Pass 1: mean (4-accumulator)
    cdef __m256 vs0 = _mm256_setzero_ps()
    cdef __m256 vs1 = _mm256_setzero_ps()
    cdef __m256 vs2 = _mm256_setzero_ps()
    cdef __m256 vs3 = _mm256_setzero_ps()
    for i in range(0, end32, 32):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[i]))
        vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&data[i + 8]))
        vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(&data[i + 16]))
        vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(&data[i + 24]))
    for i in range(end32, end8, 8):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[i]))
    vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
    cdef float mean = _hsum_avx(vs0)
    for i in range(end8, n):
        mean += data[i]
    mean /= n

    # Pass 2: variance (FMA)
    cdef __m256 vmean = _mm256_set1_ps(mean)
    cdef __m256 vd
    vs0 = _mm256_setzero_ps()
    vs1 = _mm256_setzero_ps()
    vs2 = _mm256_setzero_ps()
    vs3 = _mm256_setzero_ps()
    for i in range(0, end32, 32):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i]), vmean)
        vs0 = _mm256_fmadd_ps(vd, vd, vs0)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i + 8]), vmean)
        vs1 = _mm256_fmadd_ps(vd, vd, vs1)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i + 16]), vmean)
        vs2 = _mm256_fmadd_ps(vd, vd, vs2)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i + 24]), vmean)
        vs3 = _mm256_fmadd_ps(vd, vd, vs3)
    for i in range(end32, end8, 8):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i]), vmean)
        vs0 = _mm256_fmadd_ps(vd, vd, vs0)
    vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
    cdef float var_sum = _hsum_avx(vs0)
    for i in range(end8, n):
        var_sum += (data[i] - mean) * (data[i] - mean)

    # Pass 3: normalize with AVX (gamma=1, beta=0)
    cdef float inv_std = 1.0 / sqrtf(var_sum / n + 1e-5)
    cdef __m256 vinv = _mm256_set1_ps(inv_std)
    for i in range(0, end8, 8):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&data[i]), vmean)
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(vd, vinv))
    for i in range(end8, n):
        out[i] = (data[i] - mean) * inv_std

    # Reduce output
    vs0 = _mm256_setzero_ps()
    vs1 = _mm256_setzero_ps()
    for i in range(0, (n // 16) * 16, 16):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&out[i]))
        vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&out[i + 8]))
    for i in range((n // 16) * 16, end8, 8):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&out[i]))
    vs0 = _mm256_add_ps(vs0, vs1)
    cdef double total = <double>_hsum_avx(vs0)
    for i in range(end8, n):
        total += out[i]

    free(data)
    free(out)
    return total
