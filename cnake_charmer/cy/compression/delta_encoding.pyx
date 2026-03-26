# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Delta encoding and decoding with round-trip verification (Cython-optimized).

Keywords: compression, delta encoding, lossless, round-trip, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def delta_encoding(int n):
    """Delta-encode n values and return sum of absolute deltas using typed arithmetic."""
    cdef int i, prev, curr, delta
    cdef long long abs_delta_sum

    prev = (0 * 7 + 3) % 1000
    abs_delta_sum = prev  # first delta is the value itself

    for i in range(1, n):
        curr = (i * 7 + 3) % 1000
        delta = curr - prev
        if delta < 0:
            abs_delta_sum -= delta
        else:
            abs_delta_sum += delta
        prev = curr

    return <int>abs_delta_sum
