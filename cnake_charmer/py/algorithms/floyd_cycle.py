"""Detect cycle length using Floyd's tortoise and hare algorithm.

Keywords: algorithms, floyd, cycle detection, tortoise hare, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def floyd_cycle(n: int) -> int:
    """Detect cycle length in sequence f(x) = (x*x + 1) % n starting from x=2.

    Uses Floyd's tortoise and hare algorithm.

    Args:
        n: Modulus for the sequence.

    Returns:
        Length of the detected cycle.
    """
    # Phase 1: find meeting point
    tortoise = (2 * 2 + 1) % n
    hare = (tortoise * tortoise + 1) % n
    while tortoise != hare:
        tortoise = (tortoise * tortoise + 1) % n
        hare = (hare * hare + 1) % n
        hare = (hare * hare + 1) % n

    # Phase 2: find cycle length
    cycle_len = 1
    hare = (tortoise * tortoise + 1) % n
    while tortoise != hare:
        hare = (hare * hare + 1) % n
        cycle_len += 1

    return cycle_len
