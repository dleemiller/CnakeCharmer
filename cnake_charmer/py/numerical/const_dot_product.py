"""
Compute the dot product of two deterministic vectors.

Keywords: numerical, dot product, linear algebra, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def const_dot_product(n: int) -> float:
    """Compute dot product of two deterministic vectors.

    Vector a: a[i] = ((i * 31 + 7) % 500) / 25.0
    Vector b: b[i] = ((i * 53 + 11) % 500) / 25.0

    Args:
        n: Length of the vectors.

    Returns:
        Dot product as a float.
    """
    result = 0.0
    for i in range(n):
        a_val = ((i * 31 + 7) % 500) / 25.0
        b_val = ((i * 53 + 11) % 500) / 25.0
        result += a_val * b_val

    return result
