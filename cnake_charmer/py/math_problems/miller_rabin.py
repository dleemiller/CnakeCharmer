"""
Count primes up to n using deterministic Miller-Rabin primality test.

Keywords: math, primes, miller-rabin, primality, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def miller_rabin(n: int) -> int:
    """Count primes up to n using deterministic Miller-Rabin.

    Uses witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    which is deterministic for n < 3.3 * 10^24.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Number of primes up to n.
    """
    if n < 2:
        return 0

    def is_prime(num):
        if num < 2:
            return False
        if num < 4:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False

        # Write num-1 as 2^r * d
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
    for i in range(2, n + 1):
        if is_prime(i):
            count += 1

    return count
