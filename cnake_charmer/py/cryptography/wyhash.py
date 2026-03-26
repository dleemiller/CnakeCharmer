"""
wyhash fast hash function by Wang Yi.

Keywords: cryptography, hash, wyhash, mixing, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MASK = (1 << 64) - 1

_wyp0 = 0xA0761D6478BD642F
_wyp1 = 0xE7037ED1A0B428DB
_wyp2 = 0x8EBC6AF09C88C6E3
_wyp3 = 0x589965CC75374CC3


def _wymix(a, b):
    """128-bit multiply then xor-fold to 64 bits."""
    full = a * b
    lo = full & MASK
    hi = (full >> 64) & MASK
    return lo ^ hi


def _wyread4(data, offset):
    """Read 4 bytes little-endian."""
    return (
        data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)
    )


def _wyread8(data, offset):
    """Read 8 bytes little-endian."""
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
        | (data[offset + 4] << 32)
        | (data[offset + 5] << 40)
        | (data[offset + 6] << 48)
        | (data[offset + 7] << 56)
    )


def _wyhash(data, seed):
    """Hash a bytes-like object using wyhash."""
    length = len(data)
    seed = (seed ^ _wyp0) & MASK

    if length <= 16:
        if length >= 4:
            a = (_wyread4(data, 0) | (_wyread4(data, (length >> 3) << 2) << 32)) & MASK
            b = (
                _wyread4(data, length - 4)
                | (_wyread4(data, length - 4 - ((length >> 3) << 2)) << 32)
            ) & MASK
        elif length > 0:
            a = data[0] | (data[length >> 1] << 8) | (data[length - 1] << 16) | (length << 24)
            b = 0
        else:
            a = 0
            b = 0
        return _wymix((a ^ _wyp0) & MASK, (b ^ _wyp1) & MASK) ^ ((seed ^ length) & MASK)

    # For longer keys: process 16-byte chunks
    idx = 0
    see1 = seed
    if length > 48:
        see2 = seed
        while idx + 48 <= length:
            seed = _wymix(
                (_wyread8(data, idx) ^ _wyp0) & MASK, (_wyread8(data, idx + 8) ^ seed) & MASK
            )
            see1 = _wymix(
                (_wyread8(data, idx + 16) ^ _wyp1) & MASK, (_wyread8(data, idx + 24) ^ see1) & MASK
            )
            see2 = _wymix(
                (_wyread8(data, idx + 32) ^ _wyp2) & MASK, (_wyread8(data, idx + 40) ^ see2) & MASK
            )
            idx += 48
        seed = (seed ^ see1 ^ see2) & MASK

    while idx + 16 <= length:
        seed = _wymix((_wyread8(data, idx) ^ _wyp0) & MASK, (_wyread8(data, idx + 8) ^ seed) & MASK)
        idx += 16

    a = _wyread8(data, length - 16)
    b = _wyread8(data, length - 8)
    return _wymix((a ^ _wyp1) & MASK, (b ^ seed) & MASK) ^ (length & MASK)


@python_benchmark(args=(200000,))
def wyhash(n: int) -> tuple:
    """Hash n different byte sequences using wyhash and return discriminating results.

    Generates byte sequences deterministically and hashes each one,
    accumulating an xor of all hashes, tracking the last hash, and
    counting hashes with the high bit set.

    Args:
        n: Number of byte sequences to hash.

    Returns:
        Tuple of (total_hash_xor, last_hash, count_with_high_bit_set).
    """
    total_xor = 0
    last_hash = 0
    high_bit_count = 0

    for i in range(n):
        # Generate a deterministic byte sequence of varying length
        length = (i % 31) + 1
        data = bytes([(i * 7 + j * 13 + 5) & 0xFF for j in range(length)])
        h = _wyhash(data, 42)
        total_xor ^= h
        last_hash = h
        if h & (1 << 63):
            high_bit_count += 1

    return (total_xor, last_hash, high_bit_count)
