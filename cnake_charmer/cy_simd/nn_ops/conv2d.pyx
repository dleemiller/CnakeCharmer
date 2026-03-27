# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D convolution — AVX2+FMA broadcast kernel pattern.

Keywords: convolution, 2d, image, neural network, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
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


@cython_benchmark(syntax="cy_simd", args=(500,))
def conv2d(int n):
    """f32 2D conv with AVX2+FMA — broadcast kernel, 8 output cols at a time."""
    cdef float kern[9]
    kern[0] = -1; kern[1] = -1; kern[2] = -1
    kern[3] = -1; kern[4] = 8;  kern[5] = -1
    kern[6] = -1; kern[7] = -1; kern[8] = -1
    cdef int kh = 3, kw = 3
    cdef int oh = n - kh + 1
    cdef int ow = n - kw + 1

    cdef float *inp = <float *>malloc(n * n * sizeof(float))
    cdef float *out = <float *>malloc(oh * ow * sizeof(float))
    if not inp or not out:
        raise MemoryError()

    # Input gen: same pixel formula as py/cy
    cdef int i, j, ki, kj
    for i in range(n):
        for j in range(n):
            inp[i * n + j] = <float>((i * 17 + j * 31 + 5) % 256)

    # Conv2d: broadcast kernel tap, FMA 8 output columns at a time
    cdef __m256 vacc, vk
    cdef float s

    for i in range(oh):
        j = 0
        while j + 8 <= ow:
            vacc = _mm256_setzero_ps()
            for ki in range(kh):
                for kj in range(kw):
                    vk = _mm256_broadcast_ss(&kern[ki * kw + kj])
                    vacc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(&inp[(i + ki) * n + j + kj]), vacc)
            _mm256_storeu_ps(&out[i * ow + j], vacc)
            j += 8
        while j < ow:
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += inp[(i + ki) * n + (j + kj)] * kern[ki * kw + kj]
            out[i * ow + j] = s
            j += 1

    # Reduce with AVX
    cdef int total_out = oh * ow
    cdef int rend8 = (total_out // 8) * 8
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    cdef int rend16 = (total_out // 16) * 16
    for i in range(0, rend16, 16):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&out[i + 8]))
    for i in range(rend16, rend8, 8):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
    acc0 = _mm256_add_ps(acc0, acc1)
    cdef double total = <double>_hsum_avx(acc0)
    for i in range(rend8, total_out):
        total += out[i]

    free(inp)
    free(out)
    return total
