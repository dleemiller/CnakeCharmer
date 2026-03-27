"""Exponentially weighted moving average using NumPy arrays.

Computes EWMA with alpha=0.01 over a random array. EWMA has a
sequential data dependency so NumPy cannot vectorize the loop.

Keywords: dsp, ewma, moving average, numpy, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def numpy_ewma(n: int) -> float:
    """Compute EWMA over a random array and return the last value.

    Args:
        n: Length of the input array.

    Returns:
        Last value of the EWMA output.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    alpha = 0.01
    result = 0.0
    for i in range(n):
        result = alpha * data[i] + (1.0 - alpha) * result
    return result
