# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Exponential moving average of a deterministic signal (Cython-optimized).

Keywords: statistics, exponential moving average, EMA, signal processing, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def exponential_moving_average(int n):
    """Compute EMA of a deterministic signal and return three sample values.

    Args:
        n: Length of the signal.

    Returns:
        Tuple of (e[n-1], e[n//2], e[n//4]) — three independent EMA samples.
    """
    cdef int i, window, quarter, half
    cdef double w, one_minus_w
    cdef double ema, quarter_val, half_val
    cdef double *values = <double *>malloc(n * sizeof(double))
    if not values:
        raise MemoryError()

    window = n // 10
    if window < 2:
        window = 2
    w = 2.0 / (window + 1)
    one_minus_w = 1.0 - w
    quarter = n // 4
    half = n // 2

    with nogil:
        for i in range(n):
            values[i] = <double>((i * 7919 + 12345) % 10000)

        ema = values[0]
        quarter_val = ema
        half_val = ema

        for i in range(1, n):
            ema = values[i] * w + ema * one_minus_w
            if i == quarter:
                quarter_val = ema
            if i == half:
                half_val = ema

    free(values)
    return (ema, half_val, quarter_val)

