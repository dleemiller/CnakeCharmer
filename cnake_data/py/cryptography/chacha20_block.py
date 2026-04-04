"""Generate ChaCha20 quarter-round outputs.

Keywords: cryptography, chacha20, quarter-round, stream cipher, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF


@python_benchmark(args=(5000,))
def chacha20_block(n: int) -> int:
    """Generate n ChaCha20 blocks and return sum of first word per block.

    State: s[i] = (i*7+3) for i in 0..15. Each block applies 20 rounds
    (10 double-rounds of column + diagonal quarter-rounds).

    Args:
        n: Number of blocks to generate.

    Returns:
        Sum of first output word from each block.
    """

    def quarter_round(s, a, b, c, d):
        s[a] = (s[a] + s[b]) & MASK32
        s[d] ^= s[a]
        s[d] = ((s[d] << 16) | (s[d] >> 16)) & MASK32
        s[c] = (s[c] + s[d]) & MASK32
        s[b] ^= s[c]
        s[b] = ((s[b] << 12) | (s[b] >> 20)) & MASK32
        s[a] = (s[a] + s[b]) & MASK32
        s[d] ^= s[a]
        s[d] = ((s[d] << 8) | (s[d] >> 24)) & MASK32
        s[c] = (s[c] + s[d]) & MASK32
        s[b] ^= s[c]
        s[b] = ((s[b] << 7) | (s[b] >> 25)) & MASK32

    total = 0
    for block in range(n):
        # Initialize state per block (vary by block number)
        state = [(i * 7 + 3 + block) & MASK32 for i in range(16)]

        # 20 rounds = 10 double-rounds
        for _ in range(10):
            # Column rounds
            quarter_round(state, 0, 4, 8, 12)
            quarter_round(state, 1, 5, 9, 13)
            quarter_round(state, 2, 6, 10, 14)
            quarter_round(state, 3, 7, 11, 15)
            # Diagonal rounds
            quarter_round(state, 0, 5, 10, 15)
            quarter_round(state, 1, 6, 11, 12)
            quarter_round(state, 2, 7, 8, 13)
            quarter_round(state, 3, 4, 9, 14)

        total += state[0]

    return total
