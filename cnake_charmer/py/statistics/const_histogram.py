"""
Build a histogram from deterministic integer data.

Keywords: statistics, histogram, frequency, distribution, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def const_histogram(n: int) -> int:
    """Build histogram of values in [0, 100) and return the max bin count.

    Data: data[i] = (i * 83 + 19) % 100.

    Args:
        n: Number of data points.

    Returns:
        Maximum count in any histogram bin.
    """
    num_bins = 100
    bins = [0] * num_bins

    for i in range(n):
        val = (i * 83 + 19) % 100
        bins[val] += 1

    max_count = 0
    for i in range(num_bins):
        if bins[i] > max_count:
            max_count = bins[i]

    return max_count
