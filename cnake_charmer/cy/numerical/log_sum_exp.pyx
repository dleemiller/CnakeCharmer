# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Numerically stable log-sum-exp using the max trick (Cython-optimized).

Keywords: numerical, log_sum_exp, exponential, logarithm, stability, cython, benchmark
"""

from libc.math cimport log, exp
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def log_sum_exp(int n):
    """Compute log-sum-exp over deterministic log-probability arrays using C math."""
    cdef int i, offset
    cdef double max_lnp, sum_exp, result, accum, max_val
    cdef double *lnp = <double *>malloc(n * sizeof(double))
    if not lnp:
        raise MemoryError()

    accum = 0.0
    max_val = 0.0

    for offset in range(5):
        # Generate deterministic log-probabilities
        for i in range(n):
            lnp[i] = -((i * 7 + 3 + offset * 13) % 100) / 10.0

        with nogil:
            # Find max for numerical stability
            max_lnp = -1e308
            for i in range(n):
                if lnp[i] > max_lnp:
                    max_lnp = lnp[i]

            # Compute sum of exp(lnp - max)
            sum_exp = 0.0
            for i in range(n):
                sum_exp += exp(lnp[i] - max_lnp)

            result = log(sum_exp) + max_lnp

        accum += result
        max_val = max_lnp

    free(lnp)
    return (accum, max_val)
