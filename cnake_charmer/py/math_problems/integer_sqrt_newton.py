"""Integer square root by Newton iteration.

Keywords: math, integer sqrt, newton, loop, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _isqrt(k: int) -> int:
    x = k
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + k // x) // 2
    return x


@python_benchmark(args=(1, 7000))
def integer_sqrt_newton(start: int, stop: int) -> int:
    """Sum integer square roots for range(start, stop)."""
    total = 0
    if start < 1:
        start = 1
    for k in range(start, stop):
        total += _isqrt(k)
    return total
