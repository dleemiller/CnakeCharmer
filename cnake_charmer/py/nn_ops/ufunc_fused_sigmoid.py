"""Sigmoid activation using np.vectorize (Python-level per-element).

Python equivalent of a Cython fused-type ufunc sigmoid.
Uses np.vectorize to show the baseline for custom element-wise ops.

Keywords: nn_ops, sigmoid, activation, numpy, ufunc, fused type, benchmark
"""

import math

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


_sigmoid_vec = np.vectorize(_sigmoid)


@python_benchmark(args=(1000000,))
def ufunc_fused_sigmoid(n: int) -> float:
    """Apply sigmoid to n standard-normal values and return sum.

    Args:
        n: Number of elements.

    Returns:
        Sum of sigmoid outputs.
    """
    rng = np.random.RandomState(42)
    arr = rng.standard_normal(n)
    result = _sigmoid_vec(arr)
    return float(np.sum(result))
