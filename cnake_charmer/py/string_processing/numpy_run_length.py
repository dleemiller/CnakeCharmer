"""Run-length encoding of a 1D integer NumPy array.

Counts the number of runs of consecutive equal values.

Keywords: string_processing, run length, encoding, numpy, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def numpy_run_length(n: int) -> int:
    """Count runs of consecutive equal values.

    Args:
        n: Length of the input array.

    Returns:
        Number of runs.
    """
    rng = np.random.RandomState(42)
    data = rng.randint(0, 10, size=n)
    runs = 1
    for i in range(1, n):
        if data[i] != data[i - 1]:
            runs += 1
    return runs
