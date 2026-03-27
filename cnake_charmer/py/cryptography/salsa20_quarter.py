"""Salsa20 quarter-round applied iteratively.

Keywords: cryptography, salsa20, quarter round, stream cipher, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF


@python_benchmark(args=(500000,))
def salsa20_quarter(n: int) -> tuple:
    """Apply Salsa20 quarter-round n times to a 16-word state.

    Initializes state from fixed seed, then repeatedly applies
    quarter-round on rotating sets of 4 state words.

    Args:
        n: Number of quarter-round applications.

    Returns:
        Tuple of (state_sum, first_word, last_word).
    """
    # Initialize 16-word state
    state = [0] * 16
    for i in range(16):
        state[i] = (i * 0x9E3779B9 + 0x12345678) & MASK32

    mask = MASK32

    for step in range(n):
        # Pick 4 indices based on step
        base = (step * 4) % 16
        a = base
        b = (base + 1) % 16
        c = (base + 2) % 16
        d = (base + 3) % 16

        # Quarter-round: a, b, c, d
        t = (state[a] + state[d]) & mask
        state[b] ^= ((t << 7) | (t >> 25)) & mask
        t = (state[b] + state[a]) & mask
        state[c] ^= ((t << 9) | (t >> 23)) & mask
        t = (state[c] + state[b]) & mask
        state[d] ^= ((t << 13) | (t >> 19)) & mask
        t = (state[d] + state[c]) & mask
        state[a] ^= ((t << 18) | (t >> 14)) & mask

    state_sum = 0
    for i in range(16):
        state_sum = (state_sum + state[i]) & MASK32
    first_word = state[0]
    last_word = state[15]

    return (state_sum, first_word, last_word)
