"""Deterministic dropout mask on a float tensor.

Apply dropout with deterministic mask: mask[i] = 1 if (i*7+3)%100 >= p*100 else 0.
output[i] = input[i] * mask[i] / (1 - p).

Keywords: dropout, mask, neural network, tensor, f32, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def dropout_mask(n: int) -> float:
    """Apply deterministic dropout mask with p=0.1, return sum.

    Args:
        n: Tensor size (number of float elements).

    Returns:
        Sum of masked values.
    """
    p = 0.1
    scale = 1.0 / (1.0 - p)
    threshold = int(p * 100)

    # Generate input
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # Apply dropout mask
    total = 0.0
    for i in range(n):
        if (i * 7 + 3) % 100 >= threshold:
            total += data[i] * scale
        # else: output is 0, no contribution

    return total
