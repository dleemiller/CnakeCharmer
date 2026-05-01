"""Number-theory helpers: gcd/lcm, primality, Miller-Rabin, and prime generation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import reduce


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def lcm(*args: int) -> int:
    return reduce(_lcm, args)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def modular_exponentiation(a: int, b: int, n: int) -> int:
    d = 1
    for bit in bin(b)[2:]:
        d = (d * d) % n
        if bit == "1":
            d = (d * a) % n
    return d


def witness(a: int, n: int) -> bool:
    t = 0
    u = n - 1
    while (u & 1) == 0:
        t += 1
        u >>= 1
    x0 = modular_exponentiation(a, u, n)
    x1 = x0
    for _ in range(1, t + 1):
        x1 = (x0 * x0) % n
        if x1 == 1 and x0 not in (1, n - 1):
            return True
        x0 = x1
    return x1 != 1


def miller_rabin(n: int, s: int = 8) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    for _ in range(s):
        a = random.randint(1, n - 1)
        if witness(a, n):
            return False
    return True


def prime_factorize(number: int) -> dict[int, int]:
    factors = defaultdict(int)
    divider = 2
    n = number
    while n > 1:
        if n % divider == 0:
            n //= divider
            factors[divider] += 1
        else:
            divider += 1
    return dict(factors)


def primes(n: int) -> list[int]:
    if n <= 2:
        return []
    sieve = [True] * (n + 1)
    for x in range(3, int(n**0.5) + 1, 2):
        for y in range(3, (n // x) + 1, 2):
            sieve[x * y] = False
    return [2] + [i for i in range(3, n, 2) if sieve[i]]
