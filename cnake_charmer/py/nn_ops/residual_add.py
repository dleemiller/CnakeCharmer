"""Fused residual add + ReLU.

output = relu(input + residual).

Keywords: residual, add, relu, neural network, tensor, f32, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def residual_add(n: int) -> float:
    """Fused residual add + ReLU, return sum.

    Args:
        n: Tensor size (number of float elements).

    Returns:
        Sum of output values.
    """
    # Generate input and residual
    inp = [math.sin(i * 0.01) * 10.0 for i in range(n)]
    residual = [math.cos(i * 0.01) * 10.0 for i in range(n)]

    # Fused residual add + ReLU
    total = 0.0
    for i in range(n):
        val = inp[i] + residual[i]
        if val < 0.0:
            val = 0.0
        total += val

    return total
