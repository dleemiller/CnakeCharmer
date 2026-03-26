"""
Deterministic reservoir sampling using LCG pseudo-random numbers.

Keywords: algorithms, reservoir sampling, random, LCG, streaming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def reservoir_sampling(n: int) -> int:
    """Deterministic reservoir sampling over n items, return sum of reservoir.

    Uses a Linear Congruential Generator (LCG) for reproducible randomness.
    Reservoir size k=100. Stream values are v[i] = (i*31+17) % 1000000.

    Args:
        n: Number of stream items.

    Returns:
        Sum of the k=100 items in the final reservoir.
    """
    k = 100
    # LCG parameters (glibc constants)
    a = 1103515245
    c = 12345
    m = 2147483648  # 2^31
    seed = 42

    # Fill reservoir with first k items
    reservoir = [0] * k
    for i in range(k):
        reservoir[i] = (i * 31 + 17) % 1000000

    # Process remaining items
    rng = seed
    for i in range(k, n):
        rng = (a * rng + c) % m
        j = rng % (i + 1)
        if j < k:
            reservoir[j] = (i * 31 + 17) % 1000000

    total = 0
    for i in range(k):
        total += reservoir[i]
    return total
