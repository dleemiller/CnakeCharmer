# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GELU on f32 tensor -- AVX2 where possible, scalar tanh fallback.

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

Keywords: gelu, activation, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, tanh
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()

cdef float SQRT_2_OVER_PI = 0.7978845608028654
cdef float GELU_COEFF = 0.044715


@cython_benchmark(syntax="cy_simd", args=(5000000,))
def gelu(int n):
    """f32 GELU -- scalar (tanh not vectorizable), AVX reduction."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float x, inner

    # Allocate tensor
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # GELU in-place (scalar -- tanh has no AVX intrinsic)
    for i in range(n):
        x = data[i]
        inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
        data[i] = 0.5 * x * (1.0 + tanh(inner))

    # AVX reduction
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int end8 = (n // 8) * 8
    cdef float tmp[8]

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[i]))
    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, n):
        total += data[i]

    free(data)
    return total
