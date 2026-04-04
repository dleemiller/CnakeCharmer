"""Compute hash-based checksum from rapidly created Pair objects.

Keywords: algorithms, pair, freelist, extension type, hashing, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Pair:
    """Simple key-value pair."""

    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def hash_code(self):
        return ((self.key * 2654435761) ^ (self.value * 1664525)) & 0xFFFFFFFF


@python_benchmark(args=(100000,))
def freelist_pair_hash(n: int) -> int:
    """Create n pairs, compute cumulative hash checksum.

    Args:
        n: Number of pairs to create.

    Returns:
        Cumulative XOR checksum of all pair hashes.
    """
    checksum = 0
    for i in range(n):
        key = (i * 1103515245 + 12345) & 0xFFFFFFFF
        value = (i * 214013 + 2531011) & 0xFFFFFFFF
        p = Pair(key, value)
        checksum ^= p.hash_code()
    return checksum
