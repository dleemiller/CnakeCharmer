# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Salsa20 quarter-round applied iteratively (Cython-optimized).

Keywords: cryptography, salsa20, quarter round, stream cipher, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def salsa20_quarter(int n):
    """Apply Salsa20 quarter-round n times to a 16-word state."""
    cdef unsigned int state[16]
    cdef unsigned int t
    cdef int step, a, b, c, d, base, i
    cdef unsigned int state_sum, first_word, last_word

    # Initialize 16-word state
    for i in range(16):
        state[i] = <unsigned int>i * <unsigned int>0x9E3779B9 + <unsigned int>0x12345678

    with nogil:
        for step in range(n):
            base = (step * 4) % 16
            a = base
            b = (base + 1) % 16
            c = (base + 2) % 16
            d = (base + 3) % 16

            # Quarter-round
            t = state[a] + state[d]
            state[b] = state[b] ^ ((t << 7) | (t >> 25))
            t = state[b] + state[a]
            state[c] = state[c] ^ ((t << 9) | (t >> 23))
            t = state[c] + state[b]
            state[d] = state[d] ^ ((t << 13) | (t >> 19))
            t = state[d] + state[c]
            state[a] = state[a] ^ ((t << 18) | (t >> 14))

    state_sum = 0
    for i in range(16):
        state_sum = state_sum + state[i]
    first_word = state[0]
    last_word = state[15]

    return (int(state_sum), int(first_word), int(last_word))
