"""Queue-based coin-sum combinations and simple prime list."""

from __future__ import annotations

import math


def find_combi_iterative(amount, coins, sr, lr):
    queue = [[amount, 0, 0]]
    result = 0

    while len(queue) != 0:
        c_amount, c_index, c_level = queue.pop(0)
        for i in range(c_index, len(coins)):
            n_amount = c_amount - coins[i]
            n_level = c_level + 1
            if n_amount > 0 and n_level < lr:
                queue.append([n_amount, i, n_level])
            if n_amount == 0 and n_level >= sr:
                result += 1
    return result


def test_prime(n):
    if n == 1 or n == 2:
        return False
    maxdivisor = int(math.floor(math.sqrt(n)))
    for i in range(3, maxdivisor + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_prime_list(n):
    prime_list = [1, 2]
    for i in range(3, n, 2):
        if test_prime(i):
            prime_list.append(i)
    prime_list.append(n)
    return prime_list
