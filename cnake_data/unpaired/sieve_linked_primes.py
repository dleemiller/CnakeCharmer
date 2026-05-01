"""Prime sieve using linked-list style active-node elimination."""

from __future__ import annotations


def sieve(n: int) -> list[int]:
    if n < 2:
        return []

    alive = [True] * (n + 1)
    alive[0] = False
    alive[1] = False

    i = 2
    while i * i <= n:
        if alive[i]:
            j = i * i
            while j <= n:
                alive[j] = False
                j += i
        i += 1

    return [k for k in range(2, n + 1) if alive[k]]
