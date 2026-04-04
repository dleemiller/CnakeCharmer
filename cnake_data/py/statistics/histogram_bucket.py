"""Build a histogram and compute statistics from buckets.

Keywords: histogram, bucket, statistics, container, sequence protocol, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def histogram_bucket(n: int) -> tuple:
    """Generate n values, bucket them into a 256-bin histogram, compute stats.

    Returns (max_count, num_nonempty_bins, entropy_x1000).

    Args:
        n: Number of values to histogram.

    Returns:
        Tuple of (max_count, num_nonempty_bins, entropy_x1000).
    """
    import math

    num_bins = 256
    counts = [0] * num_bins
    size = num_bins

    # Fill histogram
    for i in range(n):
        h = ((i * 2654435761 + 7) >> 4) & 0xFF
        counts[h] += 1

    # Compute stats by reading bins
    max_count = 0
    nonempty = 0
    for i in range(size):
        c = counts[i]
        if c > max_count:
            max_count = c
        if c > 0:
            nonempty += 1

    # Entropy
    entropy = 0.0
    for i in range(size):
        if counts[i] > 0:
            p = counts[i] / n
            entropy -= p * math.log(p)

    return (max_count, nonempty, int(entropy * 1000))
