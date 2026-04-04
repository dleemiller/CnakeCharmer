"""Count set bits using a bit array container.

Keywords: bit array, bitset, popcount, container, sequence protocol, algorithms, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def bit_array_count(n: int) -> int:
    """Create a bit array of n bits, set bits at deterministic positions, count set bits.

    Also tests membership (contains) for n/2 random positions and counts hits.
    Returns popcount * 1000 + membership_hits.

    Args:
        n: Number of bits.

    Returns:
        Combined result: popcount * 1000 + membership_hits.
    """
    # Bit array using list of ints (32 bits each)
    num_words = (n + 31) // 32
    words = [0] * num_words

    # Set bits
    for i in range(n):
        h = ((i * 2654435761 + 13) >> 4) & 0xFF
        if h < 77:  # ~30% density
            word_idx = i // 32
            bit_idx = i % 32
            words[word_idx] |= 1 << bit_idx

    # Popcount
    popcount = 0
    for w in words:
        while w:
            popcount += w & 1
            w >>= 1

    # Membership test
    hits = 0
    for i in range(n // 2):
        pos = ((i * 1103515245 + 7) >> 3) % n
        word_idx = pos // 32
        bit_idx = pos % 32
        if words[word_idx] & (1 << bit_idx):
            hits += 1

    return popcount * 1000 + hits
