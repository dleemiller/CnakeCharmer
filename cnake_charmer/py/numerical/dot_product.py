"""
Dot product of two vectors.

Keywords: dot product, vector, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def dot_product(n: int) -> float:
    """Compute the dot product of two vectors of length n.

    Creates vectors a[i] = i * 0.5 and b[i] = n - i, then computes sum(a[i]*b[i]).

    Args:
        n: Vector length.

    Returns:
        The dot product as a float.
    """
    a = [i * 0.5 for i in range(n)]
    b = [float(n - i) for i in range(n)]

    result = 0.0
    for i in range(n):
        result += a[i] * b[i]

    return result
