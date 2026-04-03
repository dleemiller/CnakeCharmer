# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Two LCG generators compete: genA uses multiplier 16807, genB uses 48271,
both mod 2147483647. Count how many times their lowest 16 bits match (Cython-optimized).

Keywords: lcg, random, generator, modular, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def lcg_judge_match(int n):
    """Run two LCG generators and count low-16-bit matches using typed C vars."""
    cdef long long a = 65
    cdef long long b = 8921
    cdef long long mod = 2147483647
    cdef long long mask = 65535
    cdef int matches = 0
    cdef int i

    with nogil:
        for i in range(n):
            a = (a * 16807) % mod
            b = (b * 48271) % mod
            if (a & mask) == (b & mask):
                matches += 1

    return (matches, <object>a, <object>b)
