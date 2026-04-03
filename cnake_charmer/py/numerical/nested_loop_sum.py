"""Nested loop accumulation.

Demonstrates Cython speedup on deeply nested loops with typed integer
multiplication and accumulation.

Keywords: nested_loop, accumulation, integer, multiplication, numerical
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def nested_loop_sum(n: int) -> int:
    """Compute sum of i*j*n over nested loops for i in range(n), j in range(100).

    Args:
        n: Outer loop bound.

    Returns:
        Total accumulated sum (modulo 2^63 to avoid overflow reporting issues).
    """
    total = 0
    for i in range(n):
        for j in range(100):
            total += i * j * n
    return total % (2**63)
