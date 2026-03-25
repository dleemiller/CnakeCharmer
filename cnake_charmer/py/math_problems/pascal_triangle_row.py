"""
Compute the nth row of Pascal's triangle.

Keywords: pascal, triangle, binomial, combinatorics, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def pascal_triangle_row(n: int) -> list:
    """Compute the nth row of Pascal's triangle using the iterative method.

    Builds the row iteratively: start with [1], then for each subsequent row
    compute each element as the sum of the two elements above it.

    Args:
        n: The row index (0-indexed) of Pascal's triangle to compute.

    Returns:
        List of ints representing the nth row of Pascal's triangle.
    """
    row = [1] * (n + 1)

    for i in range(2, n + 1):
        prev = 1
        for j in range(1, i):
            temp = row[j]
            row[j] = prev + row[j]
            prev = temp

    return row
