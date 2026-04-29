"""Generate a list of the first N primes (capped at 1000)."""

from __future__ import annotations


def primes(nb_primes: int):
    if nb_primes > 1000:
        nb_primes = 1000

    p = [0] * 1000
    len_p = 0
    n = 2
    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    return [prime for prime in p[:len_p]]
