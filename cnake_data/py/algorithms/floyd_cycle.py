"""Detect cycle lengths using Floyd's tortoise and hare algorithm.

Keywords: algorithms, floyd, cycle detection, tortoise hare, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def floyd_cycle(n: int) -> int:
    """Sum cycle lengths for n different sequences f(x) = (x*x + c) % 1000003.

    Args:
        n: Number of sequences to test (each with c = i).

    Returns:
        Sum of all cycle lengths found.
    """
    total = 0
    mod = 1000003

    for i in range(n):
        c = i + 1
        tortoise = (2 * 2 + c) % mod
        hare = (tortoise * tortoise + c) % mod
        while tortoise != hare:
            tortoise = (tortoise * tortoise + c) % mod
            hare = (hare * hare + c) % mod
            hare = (hare * hare + c) % mod

        cycle_len = 1
        hare = (tortoise * tortoise + c) % mod
        while tortoise != hare:
            hare = (hare * hare + c) % mod
            cycle_len += 1

        total += cycle_len

    return total
