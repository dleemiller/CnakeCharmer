"""Parallel partial histograms with chunk-based processing.

Keywords: statistics, histogram, parallel, chunked, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_histogram_partial(n: int) -> int:
    """Build histogram of n deterministic values into 256 bins.

    Simulates chunk-based parallel histogram building by
    processing data in chunks and merging partial results.

    Args:
        n: Number of values to bin.

    Returns:
        Maximum bin count.
    """
    num_bins = 256
    bins = [0] * num_bins

    for i in range(n):
        val = ((i * 2654435761 + 17) >> 8) & 255
        bins[val] += 1

    max_count = 0
    for c in bins:
        if c > max_count:
            max_count = c

    return max_count
