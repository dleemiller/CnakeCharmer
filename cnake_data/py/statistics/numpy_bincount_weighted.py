"""Weighted bin counting using NumPy.

Uses np.bincount with weights for fast histogram-like aggregation.

Keywords: statistics, bincount, weighted, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def numpy_bincount_weighted(n: int) -> float:
    """Weighted bin count and return max bin value.

    Args:
        n: Number of samples.

    Returns:
        Maximum bin value.
    """
    rng = np.random.RandomState(42)
    indices = rng.randint(0, 1000, size=n)
    weights = rng.random(n)
    counts = np.bincount(indices, weights=weights)
    return float(np.max(counts))
