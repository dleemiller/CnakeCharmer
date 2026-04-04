"""Softmax over chunks of a NumPy vector.

Applies softmax to non-overlapping chunks of size 256 and returns
the total sum of all softmax outputs.

Keywords: nn_ops, softmax, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def numpy_softmax(n: int) -> float:
    """Compute chunked softmax and return sum of outputs.

    Args:
        n: Length of the input vector.

    Returns:
        Sum of all softmax outputs.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    chunk = 256
    total = 0.0
    num_chunks = n // chunk
    for c in range(num_chunks):
        s = c * chunk
        e = s + chunk
        x = data[s:e]
        mx = np.max(x)
        ex = np.exp(x - mx)
        total += float(np.sum(ex / np.sum(ex)))
    return total
