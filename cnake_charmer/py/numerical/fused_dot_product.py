"""Dot product of two arrays using generic floating-point computation.

Keywords: numerical, dot product, linear algebra, fused type, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_dot_product(n: int) -> float:
    """Compute dot product of two float arrays.

    a[i] = (i * 13 + 7) % 997 / 100.0
    b[i] = (i * 19 + 3) % 991 / 100.0

    Args:
        n: Number of elements.

    Returns:
        Dot product as float.
    """
    total = 0.0
    for i in range(n):
        a_val = ((i * 13 + 7) % 997) / 100.0
        b_val = ((i * 19 + 3) % 991) / 100.0
        total += a_val * b_val

    return total
