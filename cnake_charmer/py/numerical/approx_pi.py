"""
Monte Carlo pi approximation using a fixed-seed LCG random number generator.

Keywords: monte carlo, pi, numerical, random, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def approx_pi(n: int) -> float:
    """Approximate pi using Monte Carlo method with a deterministic LCG PRNG.

    Uses a simple linear congruential generator for reproducibility:
    seed = (a * seed + c) % m, with a=1103515245, c=12345, m=2^31.

    Args:
        n: Number of random points to sample.

    Returns:
        Approximation of pi as a float.
    """
    inside = 0
    seed = 42
    a = 1103515245
    c = 12345
    m = 2147483648  # 2^31

    for _ in range(n):
        seed = (a * seed + c) % m
        x = seed / m
        seed = (a * seed + c) % m
        y = seed / m
        if x * x + y * y <= 1.0:
            inside += 1

    return 4.0 * inside / n
