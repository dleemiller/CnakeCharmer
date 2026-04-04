"""ReLU activation on a float tensor.

Simulates a real NN forward pass: allocate f32 tensor, apply ReLU
in-place, reduce. Matches XNNPACK f32-vclamp pattern.

Keywords: relu, activation, neural network, tensor, f32, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def relu(n: int) -> float:
    """Allocate f32 tensor, apply ReLU in-place, return sum.

    Args:
        n: Tensor size (number of float elements).

    Returns:
        Sum of activated values.
    """
    # Simulate receiving tensor from previous layer
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # ReLU in-place: max(0, x)
    for i in range(n):
        if data[i] < 0.0:
            data[i] = 0.0

    # Reduce
    total = 0.0
    for i in range(n):
        total += data[i]

    return total
