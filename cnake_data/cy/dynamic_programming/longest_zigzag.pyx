# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Length of longest zigzag subsequence (Cython-optimized).

Keywords: dynamic programming, zigzag, subsequence, alternating, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def longest_zigzag(int n):
    """Find longest zigzag subsequence length using typed arithmetic."""
    cdef int i, vi, vi_prev, up, down

    if n <= 0:
        return 0
    if n == 1:
        return 1

    up = 1
    down = 1

    for i in range(1, n):
        vi = (i * 31 + 17) % 1000
        vi_prev = ((i - 1) * 31 + 17) % 1000
        if vi > vi_prev:
            up = down + 1
        elif vi < vi_prev:
            down = up + 1

    if up > down:
        return up
    return down
