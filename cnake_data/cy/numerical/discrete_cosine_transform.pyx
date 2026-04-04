# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Discrete Cosine Transform (DCT-II) (Cython-optimized).

Keywords: numerical, DCT, cosine, transform, signal processing, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def discrete_cosine_transform(int n):
    """Compute the DCT-II of a deterministic signal."""
    cdef double *x = <double *>malloc(n * sizeof(double))
    if not x:
        raise MemoryError()

    cdef int i, k
    cdef double val, pi_over_n, coeff_sum, coeff_max, coeff_quarter
    cdef int quarter = n // 4

    # Build input signal
    for i in range(n):
        x[i] = sin(i * 0.05) + 0.5 * cos(i * 0.13)

    # Compute DCT-II
    pi_over_n = M_PI / n
    coeff_sum = 0.0
    coeff_max = -1e300
    coeff_quarter = 0.0

    for k in range(n):
        val = 0.0
        for i in range(n):
            val += x[i] * cos(pi_over_n * (i + 0.5) * k)
        coeff_sum += val
        if val > coeff_max:
            coeff_max = val
        if k == quarter:
            coeff_quarter = val

    free(x)
    return (coeff_sum, coeff_max, coeff_quarter)
