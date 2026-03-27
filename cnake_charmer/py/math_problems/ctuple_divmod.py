"""Compute divmod of hash-derived pairs and accumulate quotient and remainder sums.

Keywords: ctuple, divmod, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def ctuple_divmod(n: int) -> int:
    """Compute divmod of n pairs, return q_sum + r_sum.

    Args:
        n: Number of pairs to process.

    Returns:
        Sum of all quotients plus sum of all remainders.
    """
    q_sum = 0
    r_sum = 0
    for i in range(n):
        a = ((i * 2654435761) & 0xFFFFFFFF) % 1000000 + 1
        b = ((i * 2246822519) & 0xFFFFFFFF) % 999 + 1
        q = a // b
        r = a % b
        q_sum += q
        r_sum += r
    return q_sum + r_sum
