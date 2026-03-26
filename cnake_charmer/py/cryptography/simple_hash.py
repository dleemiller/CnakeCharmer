"""
FNV-1a style rolling hash computation.

Keywords: cryptography, hash, fnv, rolling, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def simple_hash(n: int) -> int:
    """Compute an FNV-1a style rolling hash over n bytes.

    Data: b[i] = (i*7+3) % 256
    FNV-1a: hash = offset_basis; for each byte: hash ^= byte; hash *= prime
    Uses 64-bit FNV parameters, masked to 64 bits.

    Args:
        n: Number of bytes to hash.

    Returns:
        Final 64-bit hash value.
    """
    FNV_OFFSET = 14695981039346656037
    FNV_PRIME = 1099511628211
    MASK = (1 << 64) - 1

    h = FNV_OFFSET
    for i in range(n):
        b = (i * 7 + 3) % 256
        h = h ^ b
        h = (h * FNV_PRIME) & MASK

    return h
