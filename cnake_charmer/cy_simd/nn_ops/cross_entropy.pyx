# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cross entropy loss on f32 tensor -- scalar (exp not vectorizable).

Keywords: cross_entropy, loss, neural network, tensor, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp, log
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_max_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(100000,))
def cross_entropy(int n):
    """f32 cross entropy loss -- scalar (exp has no AVX intrinsic)."""
    cdef float *logits = <float *>malloc(n * sizeof(float))
    if not logits:
        raise MemoryError()

    cdef int i
    cdef int target = 0
    cdef float max_logit
    cdef double sum_exp = 0.0
    cdef double loss

    # Generate logits
    for i in range(n):
        logits[i] = (i * 17 + 5) % 100 / 10.0 - 5.0

    # Find max
    max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Sum exp(logits - max)
    for i in range(n):
        sum_exp += exp(<double>(logits[i] - max_logit))

    loss = -<double>logits[target] + log(sum_exp) + <double>max_logit

    free(logits)
    return loss
