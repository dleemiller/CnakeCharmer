"""LZ4-style block compression of deterministic data using a 4096-entry hash table.

Keywords: compression, lz4, hash table, block compression, sliding window, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

_HASH_BITS = 12
_HASH_SIZE = 1 << _HASH_BITS   # 4096
_HASH_MULT = 2654435761         # Knuth multiplicative hash
_MIN_MATCH = 4
_MAX_OFFSET = 65535
_MAX_MATCH = 255 + _MIN_MATCH


@python_benchmark(args=(200000,))
def lz4_block(n: int) -> tuple:
    """Compress n bytes of deterministic data using LZ4-style block encoding.

    Data: byte i = ((i * 7 + 3) % 26 + 65) — alphabet letters with period 26,
    giving frequent back-references at ~26-byte offsets.

    Hash table maps 4-byte fingerprints to their most recent position.
    Greedy: if a match of >= 4 bytes is found, emit it; else emit literal.

    Args:
        n: Number of bytes to compress.

    Returns:
        Tuple of (literal_count, match_count, total_tokens) where
        total_tokens = literal_count + match_count.
    """
    # Generate source data
    data = bytes(((i * 7 + 3) % 26 + 65) for i in range(n))

    hash_table = [-1] * _HASH_SIZE

    pos = 0
    literal_count = 0
    match_count = 0

    while pos <= n - _MIN_MATCH:
        # Compute 4-byte hash
        val = (
            data[pos]
            | (data[pos + 1] << 8)
            | (data[pos + 2] << 16)
            | (data[pos + 3] << 24)
        )
        h = ((val * _HASH_MULT) & 0xFFFFFFFF) >> (32 - _HASH_BITS)

        candidate = hash_table[h]
        hash_table[h] = pos

        if candidate >= 0 and 0 < pos - candidate <= _MAX_OFFSET:
            # Measure match length
            match_len = 0
            max_len = min(_MAX_MATCH, n - pos, pos - candidate)
            while match_len < max_len and data[candidate + match_len] == data[pos + match_len]:
                match_len += 1

            if match_len >= _MIN_MATCH:
                match_count += 1
                pos += match_len
                continue

        literal_count += 1
        pos += 1

    # Remaining bytes past the last MIN_MATCH window are literals
    literal_count += n - pos

    return (literal_count, match_count, literal_count + match_count)
