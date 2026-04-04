"""Elementwise addition of two f32 tensors.

Keywords: elementwise, add, neural network, tensor, f32, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def elementwise_add(n: int) -> float:
    """Add two tensors element-wise and return sum.

    a[i] = ((i * 31 + 17) % 1000) * 0.01
    b[i] = ((i * 13 + 7) % 500) * 0.01

    Args:
        n: Tensor size.

    Returns:
        Sum of output tensor.
    """
    total = 0.0
    for i in range(n):
        a = ((i * 31 + 17) % 1000) * 0.01
        b = ((i * 13 + 7) % 500) * 0.01
        total += a + b
    return total
