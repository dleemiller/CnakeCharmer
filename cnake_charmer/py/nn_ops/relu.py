"""ReLU activation function on a pre-allocated tensor.

Simulates a real NN forward pass: allocate tensor, apply ReLU in-place,
reduce to verify. All three tiers (py/cy/cy_simd) operate on the same
array pattern — only the inner loop changes.

Keywords: relu, activation, neural network, elementwise, tensor, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def relu(n: int) -> int:
    """Allocate a tensor of n values, apply ReLU in-place, return sum.

    Args:
        n: Tensor size (number of elements).

    Returns:
        Sum of activated values.
    """
    # Allocate tensor (simulates receiving a tensor from previous layer)
    data = [(i * 17 + 5) % 201 - 100 for i in range(n)]

    # Apply ReLU in-place
    for i in range(n):
        if data[i] < 0:
            data[i] = 0

    # Reduce (simulates passing to next layer or loss computation)
    total = 0
    for i in range(n):
        total += data[i]

    return total
