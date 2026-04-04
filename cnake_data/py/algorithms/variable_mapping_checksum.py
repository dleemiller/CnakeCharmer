"""Build signed-variable hash mappings and aggregate clause hashes.

Keywords: algorithms, hashing, variable mapping, checksum, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF


@python_benchmark(args=(320, 1337, 240000, 5))
def variable_mapping_checksum(n_vars: int, seed: int, draws: int, clause_len: int) -> tuple:
    """Generate +/- variable mapping and summarize deterministic clause hashes."""
    mapping: dict[int, int] = {}
    x = seed & MASK32
    for v in range(1, n_vars + 1):
        x ^= (x << 13) & MASK32
        x ^= x >> 17
        x ^= (x << 5) & MASK32
        mapping[v] = x
        x ^= (x << 13) & MASK32
        x ^= x >> 17
        x ^= (x << 5) & MASK32
        mapping[-v] = x

    checksum = 0
    collisions = 0
    last_hash = 0

    for i in range(draws):
        h = 0
        for j in range(clause_len):
            idx = ((i * 1103515245 + j * 12345 + seed) & MASK32) % n_vars + 1
            sign = -1 if ((i + j) & 1) else 1
            h += mapping[sign * idx]
        if (h & 1023) == (last_hash & 1023):
            collisions += 1
        checksum = (checksum + (h & MASK32)) & MASK32
        last_hash = h

    return (checksum, collisions, last_hash & MASK32)
