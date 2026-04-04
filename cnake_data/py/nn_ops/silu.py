"""SiLU/Swish activation on a float tensor.

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).

Keywords: silu, swish, activation, neural network, tensor, f32, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def silu(n: int) -> float:
    """Allocate f32 tensor, apply SiLU, return sum.

    Args:
        n: Tensor size (number of float elements).

    Returns:
        Sum of activated values.
    """
    # Simulate receiving tensor from previous layer
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # SiLU in-place: x * sigmoid(x) = x / (1 + exp(-x))
    total = 0.0
    for i in range(n):
        x = data[i]
        total += x / (1.0 + math.exp(-x))

    return total
