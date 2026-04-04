"""Find abundant numbers up to n and compute their excess sums.

Keywords: abundant, numbers, divisors, sigma, number theory, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def abundant_numbers(n: int) -> tuple:
    """Count abundant numbers from 1 to n and sum their excesses.

    A number is abundant if the sum of its proper divisors exceeds it.
    The excess is sigma(k) - k where sigma(k) is the sum of proper divisors.

    Uses a sieve approach: for each i, add i to sigma[j] for all multiples j.

    Args:
        n: Upper bound (inclusive) for searching abundant numbers.

    Returns:
        Tuple of (count of abundant numbers, sum of excesses mod 10^9+7,
        largest abundant number found).
    """
    mod = 10**9 + 7

    # Sieve for sum of proper divisors
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(2 * i, n + 1, i):
            sigma[j] += i

    count = 0
    excess_sum = 0
    last_abundant = 0

    for k in range(2, n + 1):
        if sigma[k] > k:
            count += 1
            excess_sum = (excess_sum + sigma[k] - k) % mod
            last_abundant = k

    return (count, excess_sum, last_abundant)
