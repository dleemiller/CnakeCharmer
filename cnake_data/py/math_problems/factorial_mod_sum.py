"""Running factorial sum modulo a fixed prime.

Keywords: math, factorial, modulo, accumulation, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 1_000_000_007


@python_benchmark(args=(250000, MOD))
def factorial_mod_sum(limit: int, mod: int) -> int:
    """Return sum_{k=1..limit} k! mod mod."""
    fact = 1
    total = 0
    for i in range(1, limit + 1):
        fact = (fact * i) % mod
        total += fact
        if total >= mod:
            total -= mod
    return total
