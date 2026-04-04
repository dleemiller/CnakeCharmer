"""Sum of Euclidean GCD over synthetic integer pairs.

Keywords: math, gcd, euclidean algorithm, integer, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


@python_benchmark(args=(6789, 9876, 120000))
def euclidean_gcd_sum(seed_a: int, seed_b: int, count: int) -> int:
    """Compute sum of gcd values for generated integer pairs."""
    total = 0
    a = seed_a
    b = seed_b
    for _ in range(count):
        a = (a * 1664525 + 1013904223) & 0x7FFFFFFF
        b = (b * 1103515245 + 12345) & 0x7FFFFFFF
        total += _gcd(a, b)
    return total
