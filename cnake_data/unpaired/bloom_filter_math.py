"""Bloom filter sizing and hashing parameter helpers."""

from __future__ import annotations

import hashlib
import math
import os


def bloom_parameters(capacity, error_rate):
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not (0 < error_rate < 1):
        raise ValueError("error rate must be in (0, 1)")

    log2 = math.log(2.0)
    log_error = math.log(error_rate)
    hash_count = max(round(-log_error / log2), 1)
    bit_count = math.ceil(-hash_count / math.log(1 - math.exp(log_error / hash_count)) * capacity)
    bit_count |= 1
    return int(hash_count), int(bit_count)


def bloom_seeds(hash_count):
    acc = bytearray()
    for _ in range(hash_count):
        random_bytes = os.urandom(8)
        acc.extend(hashlib.md5(random_bytes).digest())
    return bytes(acc)
