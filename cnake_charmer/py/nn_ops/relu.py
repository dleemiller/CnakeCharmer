"""ReLU activation function.

Keywords: relu, activation, neural network, elementwise
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def relu(n: int) -> int:
    """Apply ReLU to n values and return sum of activated values.

    v[i] = (i * 17 + 5) % 201 - 100 (centered around 0).

    Args:
        n: Number of values.

    Returns:
        Sum of activated values.
    """
    total = 0
    for i in range(n):
        v = (i * 17 + 5) % 201 - 100
        if v > 0:
            total += v
    return total
