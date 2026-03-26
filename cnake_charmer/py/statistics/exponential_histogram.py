"""Compute entropy of histogram with exponential bin widths.

Keywords: histogram, entropy, exponential bins, statistics, information theory, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def exponential_histogram(n: int) -> float:
    """Compute entropy of an exponential-width histogram.

    Values v[i] = (i*17+5) % 10000. Bins: [0,1), [1,2), [2,4), [4,8), ...
    up to covering the max value. Compute Shannon entropy of the resulting
    probability distribution.

    Args:
        n: Number of values.

    Returns:
        Shannon entropy of the histogram.
    """
    max_val = 10000
    # Determine bins: [0,1), [1,2), [2,4), [4,8), ...
    bin_edges = [0, 1]
    edge = 1
    while edge < max_val:
        edge *= 2
        bin_edges.append(edge)
    num_bins = len(bin_edges) - 1

    counts = [0] * num_bins

    for i in range(n):
        v = (i * 17 + 5) % max_val
        # Find bin using binary search on edges
        lo, hi = 0, num_bins - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if v < bin_edges[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        counts[lo] += 1

    # Compute entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)

    return entropy
