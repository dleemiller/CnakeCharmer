"""Profile mixed integer tools: gcd/lcm/sieve/nth-prime and divisor counts.

Adapted from The Stack v2 Cython candidate:
- blob_id: 97834639626f5afbe569e7cf7799a8e08b01b9ea
- filename: inttools.pyx

Keywords: math_problems, gcd, lcm, sieve, nth prime, divisors
"""

from cnake_data.benchmarks import python_benchmark


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _sieve(n: int) -> list[int]:
    if n < 2:
        return []
    a = [True] * (n + 1)
    a[0] = a[1] = False
    i = 2
    while i * i <= n:
        if a[i]:
            j = i * i
            while j <= n:
                a[j] = False
                j += i
        i += 1
    return [i for i in range(2, n + 1) if a[i]]


@python_benchmark(args=(120000,))
def stack_inttools_profile(n: int) -> tuple:
    """Compute mixed number-theory stats for deterministic workloads."""
    g = 0
    lcm_acc = 1
    mod = 1_000_000_007

    for i in range(2, 2 + n // 6):
        a = i * 37 + 11
        b = i * 53 + 7
        d = _gcd(a, b)
        g ^= d
        lcm_acc = (lcm_acc * ((a // d) * b)) % mod

    primes = _sieve(n)
    nth = primes[n // 30] if len(primes) > n // 30 else (primes[-1] if primes else 0)

    div_sum = 0
    for x in range(max(2, n - 80), n):
        c = 0
        r = 1
        while r * r <= x:
            if x % r == 0:
                c += 1 if r * r == x else 2
            r += 1
        div_sum += c

    return (g & 0xFFFFFFFF, lcm_acc, nth, div_sum)
