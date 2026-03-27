"""Clamp array elements to a fixed range using np.vectorize.

Python-level per-element clamping as the baseline for a Cython ufunc.

Keywords: numerical, clamp, clip, numpy, ufunc, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


_clamp_vec = np.vectorize(_clamp)


@python_benchmark(args=(1000000,))
def ufunc_clamp(n: int) -> float:
    """Clamp n random values to [-2.5, 2.5] and return sum.

    Args:
        n: Number of elements.

    Returns:
        Sum of clamped values.
    """
    rng = np.random.RandomState(42)
    arr = rng.standard_normal(n)
    result = _clamp_vec(arr, -2.5, 2.5)
    return float(np.sum(result))
