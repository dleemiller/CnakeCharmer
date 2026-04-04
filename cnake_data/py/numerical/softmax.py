"""Compute grouped softmax and sum all outputs.

Keywords: numerical, softmax, exponential, normalization, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def softmax(n: int) -> float:
    """Compute softmax in groups of 100 and sum all outputs.

    Values: v[i] = (i*17+5)%100 / 50.0 - 1.0.
    Computes softmax within each group of 100 elements, then sums all outputs.

    Args:
        n: Total number of elements (should be divisible by 100).

    Returns:
        Sum of all softmax outputs as a float.
    """
    GROUP = 100
    values = [(i * 17 + 5) % 100 / 50.0 - 1.0 for i in range(n)]
    num_groups = n // GROUP

    total = 0.0
    for g in range(num_groups):
        start = g * GROUP
        # Find max for numerical stability
        max_val = values[start]
        for i in range(start + 1, start + GROUP):
            if values[i] > max_val:
                max_val = values[i]

        # Compute exp and sum
        exp_sum = 0.0
        for i in range(start, start + GROUP):
            exp_sum += math.exp(values[i] - max_val)

        # Compute softmax and accumulate
        for i in range(start, start + GROUP):
            total += math.exp(values[i] - max_val) / exp_sum

    return total
