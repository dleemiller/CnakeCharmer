"""Generator matching problem (Advent of Code 2017 Day 15 style).

Two generators produce sequences using linear congruential formulas.
Count how many times the lowest 16 bits match over many iterations.

Keywords: generator, linear_congruential, bit_matching, numerical
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def generator_match_count(n: int) -> int:
    """Count 16-bit matches between two LCG generators over n rounds.

    Generator A: next = (prev * 16807) % 2147483647
    Generator B: next = (prev * 48271) % 2147483647

    Args:
        n: Number of rounds.

    Returns:
        Number of times lowest 16 bits match.
    """
    a = 65
    b = 8921
    score = 0
    mask = 65535  # 2^16 - 1
    mod = 2147483647
    for _ in range(n):
        a = (a * 16807) % mod
        b = (b * 48271) % mod
        if (a & mask) == (b & mask):
            score += 1
    return score
