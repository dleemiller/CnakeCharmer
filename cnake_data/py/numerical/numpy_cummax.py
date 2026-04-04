"""Cumulative maximum of a NumPy array.

Sequential dependency prevents NumPy vectorization.

Keywords: numerical, cumulative, maximum, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def numpy_cummax(n: int) -> float:
    """Compute cumulative max and return sum of result.

    Args:
        n: Length of the input array.

    Returns:
        Sum of the cumulative maximum array.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    cummax = np.empty(n, dtype=np.float64)
    current_max = data[0]
    cummax[0] = current_max
    for i in range(1, n):
        if data[i] > current_max:
            current_max = data[i]
        cummax[i] = current_max
    return float(np.sum(cummax))
