"""
Compute sum of digit sums of all numbers from 1 to n.

Keywords: math, digits, sum, enumeration, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def digit_sum(n: int) -> int:
    """Compute sum of digit sums for all numbers from 1 to n.

    For each number i in [1, n], compute the sum of its digits,
    then sum all those digit sums.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Total sum of all digit sums.
    """
    total = 0
    for i in range(1, n + 1):
        num = i
        while num > 0:
            total += num % 10
            num //= 10
    return total
