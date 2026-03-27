"""Deterministic Miller-Rabin primality test for numbers 2..n.

Keywords: math, primes, miller-rabin, primality, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def miller_rabin(n: int) -> tuple:
    """Test primality of all numbers 2..n using deterministic Miller-Rabin.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Tuple of (prime_count, largest_prime, prime_at_midpoint).
        prime_at_midpoint is the largest prime <= n//2.
    """
    if n < 2:
        return (0, 0, 0)

    def is_prime(num):
        if num < 2:
            return False
        if num < 4:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False

        d = num - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1

        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= num:
                continue
            x = pow(a, d, num)
            if x == 1 or x == num - 1:
                continue
            found = False
            for _ in range(r - 1):
                x = pow(x, 2, num)
                if x == num - 1:
                    found = True
                    break
            if not found:
                return False
        return True

    count = 0
    largest = 0
    mid = n // 2
    prime_at_mid = 0
    for i in range(2, n + 1):
        if is_prime(i):
            count += 1
            largest = i
            if i <= mid:
                prime_at_mid = i

    return (count, largest, prime_at_mid)
