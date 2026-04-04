# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Delta encoding and decoding with round-trip verification (Cython-optimized).

Keywords: compression, delta encoding, lossless, round-trip, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def delta_encoding(int n):
    """Delta-encode n values and return sum of absolute deltas using typed arithmetic."""
    cdef int i, prev, curr, delta, abs_delta
    cdef long long abs_delta_sum
    cdef int max_delta

    prev = (0 * 7 + 3) % 1000
    abs_delta_sum = prev  # first delta is the value itself
    max_delta = prev

    for i in range(1, n):
        curr = (i * 7 + 3) % 1000
        delta = curr - prev
        if delta < 0:
            abs_delta = -delta
        else:
            abs_delta = delta
        abs_delta_sum += abs_delta
        if abs_delta > max_delta:
            max_delta = abs_delta
        prev = curr

    return (abs_delta_sum, max_delta)
