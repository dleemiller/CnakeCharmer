# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Elementwise multiplication — AVX2 batch-16.

Keywords: elementwise, multiply, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
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


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def elementwise_mul(int n):
    """f32 elementwise multiply with AVX2 — batch-16."""
    cdef float *a = <float *>malloc(n * sizeof(float))
    cdef float *b = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not a or not b or not out:
        raise MemoryError()

    cdef int i
    for i in range(n):
        a[i] = <float>((i * 31 + 17) % 1000) * 0.01
        b[i] = <float>((i * 13 + 7) % 500) * 0.01

    # Mul: batch-16
    cdef int end16 = (n // 16) * 16
    cdef int end8 = (n // 8) * 8
    for i in range(0, end16, 16):
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])))
        _mm256_storeu_ps(&out[i+8], _mm256_mul_ps(_mm256_loadu_ps(&a[i+8]), _mm256_loadu_ps(&b[i+8])))
    for i in range(end16, end8, 8):
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])))
    for i in range(end8, n):
        out[i] = a[i] * b[i]

    # Reduce
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    for i in range(0, end16, 16):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&out[i + 8]))
    for i in range(end16, end8, 8):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
    acc0 = _mm256_add_ps(acc0, acc1)
    cdef double total = <double>_hsum_avx(acc0)
    for i in range(end8, n):
        total += out[i]

    free(a); free(b); free(out)
    return total
