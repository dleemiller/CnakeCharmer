# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Deterministic reservoir sampling using LCG pseudo-random numbers (Cython-optimized).

Keywords: algorithms, reservoir sampling, random, LCG, streaming, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def reservoir_sampling(int n):
    """Deterministic reservoir sampling over n items, return sum of reservoir."""
    cdef int k = 100
    cdef long long a = 1103515245
    cdef long long c = 12345
    cdef long long m_val = 2147483648  # 2^31
    cdef long long rng = 42
    cdef int i
    cdef long long j
    cdef int reservoir[100]
    cdef long long total

    # Fill reservoir with first k items
    for i in range(k):
        reservoir[i] = (i * 31 + 17) % 1000000

    # Process remaining items
    for i in range(k, n):
        rng = (a * rng + c) % m_val
        j = rng % (i + 1)
        if j < k:
            reservoir[j] = (i * 31 + 17) % 1000000

    total = 0
    for i in range(k):
        total += reservoir[i]
    return int(total)
