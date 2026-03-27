"""Classify floats into bins using np.vectorize.

Python-level per-element bin classification as baseline for a Cython ufunc.

Keywords: statistics, bins, classification, histogram, numpy, ufunc, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark

NUM_BINS = 20


def _classify(x, num_bins):
    b = int(x * num_bins)
    if b < 0:
        return 0
    if b >= num_bins:
        return num_bins - 1
    return b


_classify_vec = np.vectorize(_classify)


@python_benchmark(args=(1000000,))
def ufunc_classify_bin(n: int) -> int:
    """Classify n random values into 20 bins and return weighted bin sum.

    Returns sum(counts[i] * (i+1)) as a reproducible hash.

    Args:
        n: Number of elements.

    Returns:
        Weighted sum of bin counts.
    """
    rng = np.random.RandomState(42)
    arr = rng.random(n)
    bins = _classify_vec(arr, NUM_BINS)
    counts = np.bincount(bins, minlength=NUM_BINS)
    return int(np.sum(counts * np.arange(1, NUM_BINS + 1)))
