# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate arithmetic coding interval narrowing (Cython-optimized).

Keywords: compression, arithmetic coding, interval, simulation, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def arithmetic_coding_sim(int n):
    """Simulate arithmetic coding with integer intervals using typed variables."""
    cdef int cum[5]
    cum[0] = 0
    cum[1] = 1
    cum[2] = 3
    cum[3] = 6
    cum[4] = 10
    cdef int total = 10

    cdef int precision = 30
    cdef long long full_range = 1LL << precision
    cdef long long half = full_range >> 1
    cdef long long quarter = full_range >> 2
    cdef long long three_quarter = 3 * quarter

    cdef long long low = 0
    cdef long long high = full_range - 1
    cdef int num_rescales = 0
    cdef long long rng
    cdef int i, sym
    cdef long long mod_val = 1000000007

    for i in range(n):
        sym = (i * 13 + 7) % 4

        rng = high - low + 1
        high = low + (rng * cum[sym + 1]) // total - 1
        low = low + (rng * cum[sym]) // total

        while True:
            if high < half:
                low = low << 1
                high = (high << 1) | 1
                num_rescales += 1
            elif low >= half:
                low = (low - half) << 1
                high = ((high - half) << 1) | 1
                num_rescales += 1
            elif low >= quarter and high < three_quarter:
                low = (low - quarter) << 1
                high = ((high - quarter) << 1) | 1
                num_rescales += 1
            else:
                break

    return (int(low % mod_val), int(high % mod_val), num_rescales)
