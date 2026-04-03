"""Accumulation loop with division.

Computes a running sum of i/200 for i in 1..n, demonstrating typed
accumulation speedup.

Keywords: accumulation, loop, division, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def accumulate_divisions(n: int) -> float:
    """Accumulate sum of i/200 for i=1..n.

    Args:
        n: Number of iterations.

    Returns:
        1 + sum(i/200 for i in 1..n).
    """
    y = 1.0
    for i in range(1, n + 1):
        y += i / 200.0
    return y
