# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Doppler-shifted frequencies for approaching sources (Cython-optimized).

Keywords: physics, doppler, frequency, sound, wave, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def doppler_shift(int n):
    """Compute Doppler-shifted frequencies with cdivision and typed loop."""
    cdef double f0 = 1000.0
    cdef double c = 343.0
    cdef double total = 0.0
    cdef double v_source, f
    cdef int i

    for i in range(n):
        v_source = (i * 7 + 3) % 340
        f = f0 * c / (c - v_source)
        total += f

    return total
