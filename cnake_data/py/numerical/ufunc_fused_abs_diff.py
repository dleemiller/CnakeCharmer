"""Absolute element-wise difference using np.vectorize.

Python-level per-element abs diff as baseline for a Cython fused-type ufunc.

Keywords: numerical, absolute difference, numpy, ufunc, fused type, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


def _abs_diff(a, b):
    d = a - b
    return d if d >= 0 else -d


_abs_diff_vec = np.vectorize(_abs_diff)


@python_benchmark(args=(1000000,))
def ufunc_fused_abs_diff(n: int) -> float:
    """Compute sum of |a[i] - b[i]| for two random float64 arrays.

    Args:
        n: Number of elements.

    Returns:
        Sum of absolute differences.
    """
    rng = np.random.RandomState(42)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    return float(np.sum(_abs_diff_vec(a, b)))
