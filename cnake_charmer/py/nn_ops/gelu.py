"""GELU activation on a float tensor.

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

Keywords: gelu, activation, neural network, tensor, f32, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def gelu(n: int) -> float:
    """Allocate f32 tensor, apply GELU, return sum.

    Args:
        n: Tensor size (number of float elements).

    Returns:
        Sum of activated values.
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    # Simulate receiving tensor from previous layer
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # GELU in-place
    total = 0.0
    for i in range(n):
        x = data[i]
        data[i] = 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))

    # Reduce
    for i in range(n):
        total += data[i]

    return total
