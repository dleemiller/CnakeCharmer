"""Histogram computation using NumPy.

Uses np.histogram to bin random data into 256 bins over [-4, 4].

Keywords: statistics, histogram, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def numpy_typed_histogram(n: int) -> int:
    """Compute histogram with 256 bins and return max count.

    Args:
        n: Number of samples.

    Returns:
        Maximum bin count.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    bins = np.linspace(-4.0, 4.0, 257)
    counts, _ = np.histogram(data, bins=bins)
    return int(np.max(counts))
