"""
Two LCG generators compete: genA uses multiplier 16807, genB uses 48271,
both mod 2147483647. Count how many times their lowest 16 bits match.

Keywords: lcg, random, generator, modular, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def lcg_judge_match(n: int) -> tuple:
    """Run two linear congruential generators and count low-16-bit matches.

    Generator A: a = (a * 16807) % 2147483647, seed 65
    Generator B: b = (b * 48271) % 2147483647, seed 8921

    Args:
        n: Number of iterations to run.

    Returns:
        Tuple of (match_count, last_a, last_b).
    """
    a = 65
    b = 8921
    mod = 2147483647
    mask = 65535
    matches = 0

    for _ in range(n):
        a = (a * 16807) % mod
        b = (b * 48271) % mod
        if (a & mask) == (b & mask):
            matches += 1

    return (matches, a, b)
