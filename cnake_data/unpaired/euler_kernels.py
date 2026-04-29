"""Selected Project Euler-style numeric kernels."""

from __future__ import annotations


def problem1(ulimit):
    total = 0
    for i in range(3, ulimit):
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total


def problem2(ulimit):
    total, i, j = 0, 1, 2
    while j < ulimit:
        i, j = j, i + j
        if i % 2 == 0:
            total += i
    return total


def problem6(n):
    sum_of_squares = 0
    sum_of_nums = 0
    for i in range(1, n + 1):
        sum_of_nums += i
        sum_of_squares += i * i
    return sum_of_nums * sum_of_nums - sum_of_squares
