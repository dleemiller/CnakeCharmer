# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of medians of a sliding window over a deterministic sequence (Cython-optimized).

Keywords: statistics, moving median, sliding window, median, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def moving_median(int n):
    """Compute sum of medians of a sliding window of size 5.

    Args:
        n: Length of the sequence.

    Returns:
        Sum of all window medians (as int).
    """
    cdef int i, j, k, key
    cdef long long total = 0
    cdef int buf[5]

    if n < 5:
        return 0

    for i in range(n - 4):
        # Collect 5 elements
        for j in range(5):
            buf[j] = ((i + j) * 13 + 7) % 1000
        # Insertion sort the 5-element buffer
        for j in range(1, 5):
            key = buf[j]
            k = j - 1
            while k >= 0 and buf[k] > key:
                buf[k + 1] = buf[k]
                k -= 1
            buf[k + 1] = key
        total += buf[2]

    return int(total)
