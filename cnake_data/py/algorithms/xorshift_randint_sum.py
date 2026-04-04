"""Xorshift32 random sequence checksum.

Keywords: algorithms, xorshift, rng, checksum, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF


@python_benchmark(args=(2463534242, 500000, 1000))
def xorshift_randint_sum(seed: int, draws: int, bucket: int) -> int:
    """Generate xorshift values and sum value % bucket."""
    x = seed & MASK32
    total = 0
    for _ in range(draws):
        x ^= (x << 13) & MASK32
        x ^= (x >> 17) & MASK32
        x ^= (x << 5) & MASK32
        x &= MASK32
        total += x % bucket
    return total
