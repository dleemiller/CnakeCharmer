"""Compute GCD of hash-derived pairs and sum all GCDs.

Keywords: gcd, math, cpdef, standalone function, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a


@python_benchmark(args=(100000,))
def cpdef_gcd_sum(n: int) -> int:
    """Compute GCD of n hash-derived pairs and sum all GCDs.

    Args:
        n: Number of pairs to process.

    Returns:
        Sum of all GCDs.
    """
    total = 0
    for i in range(n):
        a = ((i * 2654435761) & 0xFFFFFFFF) % 100000 + 1
        b = ((i * 2246822519) & 0xFFFFFFFF) % 100000 + 1
        total += gcd(a, b)
    return total
