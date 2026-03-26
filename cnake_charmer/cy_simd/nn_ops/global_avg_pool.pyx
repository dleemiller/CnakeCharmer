# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Global average pooling — AVX2 with 4-accumulator rsum per channel.

Keywords: global_avg_pool, pooling, neural network, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
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
def global_avg_pool(int n):
    """f32 global average pool with AVX2 — 4-accumulator rsum per channel."""
    cdef int channels = 64
    cdef int spatial = n // channels

    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int c, s, offset
    cdef double total = 0.0
    cdef float channel_sum
    cdef int end32 = (spatial // 32) * 32
    cdef int end8 = (spatial // 8) * 8
    cdef __m256 vs0, vs1, vs2, vs3

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    for c in range(channels):
        offset = c * spatial

        # 4-accumulator rsum
        vs0 = _mm256_setzero_ps()
        vs1 = _mm256_setzero_ps()
        vs2 = _mm256_setzero_ps()
        vs3 = _mm256_setzero_ps()
        for s in range(0, end32, 32):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[offset + s]))
            vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&data[offset + s + 8]))
            vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(&data[offset + s + 16]))
            vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(&data[offset + s + 24]))
        for s in range(end32, end8, 8):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[offset + s]))
        vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
        channel_sum = _hsum_avx(vs0)
        for s in range(end8, spatial):
            channel_sum += data[offset + s]

        total += channel_sum / spatial

    free(data)
    return total
