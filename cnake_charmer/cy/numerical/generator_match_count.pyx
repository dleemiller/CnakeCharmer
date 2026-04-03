# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generator matching problem (Cython-optimized).

Keywords: generator, linear_congruential, bit_matching, numerical, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def generator_match_count(int n):
    """Count 16-bit matches between two LCG generators over n rounds."""
    cdef long long a = 65
    cdef long long b = 8921
    cdef int score = 0
    cdef long long mask = 65535
    cdef long long mod = 2147483647
    cdef int i

    for i in range(n):
        a = (a * 16807) % mod
        b = (b * 48271) % mod
        if (a & mask) == (b & mask):
            score += 1
    return score
