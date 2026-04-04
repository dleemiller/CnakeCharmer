# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Shannon entropy computation (Cython-optimized).

Keywords: compression, entropy, shannon, information theory, cython, benchmark
"""

from libc.math cimport log
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def entropy(int n):
    """Compute Shannon entropy of a deterministic symbol sequence."""
    cdef int freq[26]
    cdef int i
    cdef double p, result

    for i in range(26):
        freq[i] = 0

    for i in range(n):
        freq[(i * 7 + 3) % 26] += 1

    result = 0.0
    for i in range(26):
        if freq[i] > 0:
            p = <double>freq[i] / <double>n
            result -= p * log(p)

    return result
