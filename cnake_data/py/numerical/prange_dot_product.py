"""Parallel dot product of two deterministic arrays.

Keywords: numerical, dot product, reduction, parallel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_dot_product(n: int) -> float:
    """Compute dot product of two hash-derived arrays.

    Args:
        n: Length of each array.

    Returns:
        Dot product as a float.
    """
    total = 0.0
    for i in range(n):
        a_i = (i * 2654435761 & 0xFFFFFFFF) / 4294967296.0
        b_i = (i * 2246822519 & 0xFFFFFFFF) / 4294967296.0
        total += a_i * b_i

    return total
