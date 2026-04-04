"""Compute prime sieve summary metrics with modular accumulation.

Keywords: math, prime sieve, gaps, modular sum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(400000, 128, 1000003))
def prime_sieve_metrics(limit: int, window: int, mod_base: int) -> tuple:
    """Sieve primes and return count, modular checksum, and max local gap."""
    if limit < 2:
        return (0, 0, 0)

    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    i = 2
    while i * i <= limit:
        if is_prime[i]:
            step = i
            j = i * i
            while j <= limit:
                is_prime[j] = False
                j += step
        i += 1

    count = 0
    checksum = 0
    max_gap = 0
    prev = 2

    for n in range(2, limit + 1):
        if is_prime[n]:
            count += 1
            checksum = (checksum + (n % mod_base) * ((n % window) + 1)) % mod_base
            gap = n - prev
            if gap > max_gap:
                max_gap = gap
            prev = n

    return (count, checksum, max_gap)
