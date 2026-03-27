"""Build histogram of n values across 1024 bins, return max count.

Keywords: statistics, histogram, calloc, zero-initialized, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def calloc_histogram(n: int) -> int:
    """Build histogram of n deterministic values into 1024 bins.

    Args:
        n: Number of values to bin.

    Returns:
        Maximum bin count.
    """
    bins = [0] * 1024

    for i in range(n):
        val = ((i * 2654435761 + 17) >> 4) & 1023
        bins[val] += 1

    max_count = 0
    for c in bins:
        if c > max_count:
            max_count = c

    return max_count
