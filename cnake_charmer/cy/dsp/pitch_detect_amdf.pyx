# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pitch detection using Average Magnitude Difference Function (Cython-optimized).

Keywords: dsp, pitch detection, amdf, signal processing, frequency, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80000,))
def pitch_detect_amdf(int n):
    """Detect pitch of a deterministic signal using AMDF with C arrays."""
    cdef double *signal = <double *>malloc(n * sizeof(double))
    if not signal:
        raise MemoryError()

    cdef double sample_rate = 8000.0
    cdef double f1 = 100.0
    cdef double f2 = 200.0
    cdef double t
    cdef int i, li, lag
    cdef double acc, diff
    cdef double two_pi = 2.0 * M_PI

    # Generate signal
    for i in range(n):
        t = i / sample_rate
        signal[i] = 0.7 * sin(two_pi * f1 * t) + 0.3 * sin(two_pi * f2 * t)

    cdef int min_lag = 20
    cdef int max_lag = 401
    if max_lag > n // 2:
        max_lag = n // 2
    cdef int num_lags = max_lag - min_lag

    cdef double *amdf = <double *>malloc(num_lags * sizeof(double))
    if not amdf:
        free(signal)
        raise MemoryError()

    cdef int window = n // 2
    if window > 2000:
        window = 2000

    for li in range(num_lags):
        lag = min_lag + li
        acc = 0.0
        for i in range(window):
            diff = signal[i] - signal[i + lag]
            if diff < 0.0:
                diff = -diff
            acc += diff
        amdf[li] = acc / window

    # Find minimum
    cdef int best_lag = min_lag
    cdef double min_val = amdf[0]
    for li in range(1, num_lags):
        if amdf[li] < min_val:
            min_val = amdf[li]
            best_lag = min_lag + li

    # Compute energy
    cdef double energy = 0.0
    for li in range(num_lags):
        energy += amdf[li] * amdf[li]

    free(signal)
    free(amdf)

    return (best_lag, min_val, energy)
