"""Prime check over array-like inputs."""

from __future__ import annotations

import math

import numpy as np


def _is_prime(n):
    if n < 2:
        return 0
    max_v = int(math.ceil(math.sqrt(n + 1)))
    for i in range(2, max_v):
        if n % i == 0:
            return 0
    return 1


def is_prime_memoryview(values):
    c_input = np.ascontiguousarray(values, dtype=np.int64)
    out = np.zeros(len(c_input), dtype=np.bool_)
    for i in range(len(c_input)):
        out[i] = bool(_is_prime(int(c_input[i])))
    return out
