# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D convolution — AVX2+FMA broadcast kernel pattern.

Keywords: convolution, 1d, signal processing, neural network, f32, simd, avx, cython
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
def conv1d(int n):
    """f32 1D conv with AVX2+FMA — broadcast kernel, 8 outputs at a time."""
    cdef float kern[7]
    kern[0] = 0.0625; kern[1] = 0.125; kern[2] = 0.1875
    kern[3] = 0.25;   kern[4] = 0.1875; kern[5] = 0.125; kern[6] = 0.0625
    cdef int k_len = 7
    cdef int out_len = n - k_len + 1

    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(out_len * sizeof(float))
    if not inp or not out:
        raise MemoryError()

    cdef int i, k
    for i in range(n):
        inp[i] = <float>sin(i * 0.01) * 100.0

    # Conv1d: broadcast kernel tap, FMA 8 outputs at a time
    cdef int end8 = (out_len // 8) * 8
    cdef __m256 vacc, vk

    for i in range(0, end8, 8):
        vacc = _mm256_setzero_ps()
        for k in range(k_len):
            vk = _mm256_broadcast_ss(&kern[k])
            vacc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(&inp[i + k]), vacc)
        _mm256_storeu_ps(&out[i], vacc)

    # Scalar remainder
    cdef float s
    for i in range(end8, out_len):
        s = 0.0
        for k in range(k_len):
            s += inp[i + k] * kern[k]
        out[i] = s

    # Reduce with AVX
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    cdef int rend16 = (out_len // 16) * 16
    cdef int rend8 = (out_len // 8) * 8
    for i in range(0, rend16, 16):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&out[i + 8]))
    for i in range(rend16, rend8, 8):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
    acc0 = _mm256_add_ps(acc0, acc1)
    cdef double total = <double>_hsum_avx(acc0)
    for i in range(rend8, out_len):
        total += out[i]

    free(inp)
    free(out)
    return total
