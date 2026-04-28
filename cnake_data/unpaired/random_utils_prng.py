"""PRNG utilities: base class, LCG, middle-square Weyl, Halton sequence."""

from __future__ import annotations

import random

import numpy as np

UINT32_MAX = (1 << 32) - 1


class PRNG:
    def randi(self) -> int:
        return 4

    def rand(self) -> float:
        return 0.4

    def randint(self, lo: int, hi: int) -> int:
        r = int(self.rand() * (hi - lo) + lo)
        while r == hi:
            r = int(self.rand() * (hi - lo) + lo)
        return r


class PyRandGen(PRNG):
    def __init__(self, seed: int | None = None):
        random.seed(seed)

    def rand(self) -> float:
        return random.random()

    def randint(self, lo: int, hi: int) -> int:
        return random.randrange(lo, hi)


class LCG(PRNG):
    def __init__(self, mod: int, a: int, c: int, seed: int):
        if seed % 2 == 0:
            raise ValueError("Seed must be odd")
        self.state = int(seed)
        self.mod = int(mod)
        self.a = int(a)
        self.c = int(c)

    def randi(self) -> int:
        self.state = (self.a * self.state + self.c) % self.mod
        return int(self.state)

    def rand(self) -> float:
        return float(self.randi()) / float(self.mod)


def lcg_randu(seed: int) -> LCG:
    return LCG(1 << 31, 65539, 0, seed)


class MiddleSquareWeylSequence(PRNG):
    def __init__(self, seed: int = 0xB5AD4ECEDA1CE2A9):
        if seed % 2 == 0:
            raise ValueError("Seed must be odd")
        self.seed = int(seed)
        self.x = 0
        self.w = 0

    def randi(self) -> int:
        self.x = (self.x * self.x) & ((1 << 64) - 1)
        self.w = (self.w + self.seed) & ((1 << 64) - 1)
        self.x = (self.x + self.w) & ((1 << 64) - 1)
        self.x = ((self.x >> 32) | ((self.x << 32) & ((1 << 64) - 1))) & ((1 << 64) - 1)
        return self.x & UINT32_MAX

    def rand(self) -> float:
        return float(self.randi()) / float(UINT32_MAX)


def vdc(n: int, base: int = 2) -> float:
    out = 0.0
    denom = 1
    while n:
        denom *= base
        n, rem = divmod(n, base)
        out += rem / float(denom)
    return out


def primes_from_2_to(n: int) -> np.ndarray:
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def halton_sequence(size: int, dim: int, start: int = 0) -> np.ndarray:
    primes = primes_from_2_to(max(1000, dim * 20))
    if len(primes) < dim:
        raise ValueError("not enough primes generated for requested dimension")
    seq = np.zeros((dim, size), dtype=np.float32)
    for d in range(dim):
        base = int(primes[d])
        for j, i in enumerate(range(start, start + size)):
            seq[d, j] = vdc(i, base)
    return seq
