"""Sigmoid activation function.

Keywords: sigmoid, activation, neural network, elementwise, exp
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def sigmoid(n: int) -> float:
    """Apply sigmoid to n values and return sum.

    v[i] = (i * 17 + 5) % 1000 / 100.0 - 5.0

    Args:
        n: Number of values.

    Returns:
        Sum of sigmoid outputs.
    """
    total = 0.0
    for i in range(n):
        v = ((i * 17 + 5) % 1000) / 100.0 - 5.0
        total += 1.0 / (1.0 + math.exp(-v))
    return total
