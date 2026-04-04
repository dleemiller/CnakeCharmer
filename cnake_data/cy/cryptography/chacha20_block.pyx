# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generate ChaCha20 quarter-round outputs (Cython-optimized).

Keywords: cryptography, chacha20, quarter-round, stream cipher, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def chacha20_block(int n):
    """Generate n ChaCha20 blocks and return sum of first word per block."""
    cdef unsigned int *state = <unsigned int *>malloc(16 * sizeof(unsigned int))
    if not state:
        raise MemoryError()

    cdef int block, rnd, i
    cdef long long total = 0

    for block in range(n):
        # Initialize state per block
        for i in range(16):
            state[i] = <unsigned int>((i * 7 + 3 + block) & 0xFFFFFFFF)

        # 20 rounds = 10 double-rounds
        for rnd in range(10):
            # Column rounds
            state[0] = state[0] + state[4]; state[12] ^= state[0]; state[12] = (state[12] << 16) | (state[12] >> 16)
            state[8] = state[8] + state[12]; state[4] ^= state[8]; state[4] = (state[4] << 12) | (state[4] >> 20)
            state[0] = state[0] + state[4]; state[12] ^= state[0]; state[12] = (state[12] << 8) | (state[12] >> 24)
            state[8] = state[8] + state[12]; state[4] ^= state[8]; state[4] = (state[4] << 7) | (state[4] >> 25)

            state[1] = state[1] + state[5]; state[13] ^= state[1]; state[13] = (state[13] << 16) | (state[13] >> 16)
            state[9] = state[9] + state[13]; state[5] ^= state[9]; state[5] = (state[5] << 12) | (state[5] >> 20)
            state[1] = state[1] + state[5]; state[13] ^= state[1]; state[13] = (state[13] << 8) | (state[13] >> 24)
            state[9] = state[9] + state[13]; state[5] ^= state[9]; state[5] = (state[5] << 7) | (state[5] >> 25)

            state[2] = state[2] + state[6]; state[14] ^= state[2]; state[14] = (state[14] << 16) | (state[14] >> 16)
            state[10] = state[10] + state[14]; state[6] ^= state[10]; state[6] = (state[6] << 12) | (state[6] >> 20)
            state[2] = state[2] + state[6]; state[14] ^= state[2]; state[14] = (state[14] << 8) | (state[14] >> 24)
            state[10] = state[10] + state[14]; state[6] ^= state[10]; state[6] = (state[6] << 7) | (state[6] >> 25)

            state[3] = state[3] + state[7]; state[15] ^= state[3]; state[15] = (state[15] << 16) | (state[15] >> 16)
            state[11] = state[11] + state[15]; state[7] ^= state[11]; state[7] = (state[7] << 12) | (state[7] >> 20)
            state[3] = state[3] + state[7]; state[15] ^= state[3]; state[15] = (state[15] << 8) | (state[15] >> 24)
            state[11] = state[11] + state[15]; state[7] ^= state[11]; state[7] = (state[7] << 7) | (state[7] >> 25)

            # Diagonal rounds
            state[0] = state[0] + state[5]; state[15] ^= state[0]; state[15] = (state[15] << 16) | (state[15] >> 16)
            state[10] = state[10] + state[15]; state[5] ^= state[10]; state[5] = (state[5] << 12) | (state[5] >> 20)
            state[0] = state[0] + state[5]; state[15] ^= state[0]; state[15] = (state[15] << 8) | (state[15] >> 24)
            state[10] = state[10] + state[15]; state[5] ^= state[10]; state[5] = (state[5] << 7) | (state[5] >> 25)

            state[1] = state[1] + state[6]; state[12] ^= state[1]; state[12] = (state[12] << 16) | (state[12] >> 16)
            state[11] = state[11] + state[12]; state[6] ^= state[11]; state[6] = (state[6] << 12) | (state[6] >> 20)
            state[1] = state[1] + state[6]; state[12] ^= state[1]; state[12] = (state[12] << 8) | (state[12] >> 24)
            state[11] = state[11] + state[12]; state[6] ^= state[11]; state[6] = (state[6] << 7) | (state[6] >> 25)

            state[2] = state[2] + state[7]; state[13] ^= state[2]; state[13] = (state[13] << 16) | (state[13] >> 16)
            state[8] = state[8] + state[13]; state[7] ^= state[8]; state[7] = (state[7] << 12) | (state[7] >> 20)
            state[2] = state[2] + state[7]; state[13] ^= state[2]; state[13] = (state[13] << 8) | (state[13] >> 24)
            state[8] = state[8] + state[13]; state[7] ^= state[8]; state[7] = (state[7] << 7) | (state[7] >> 25)

            state[3] = state[3] + state[4]; state[14] ^= state[3]; state[14] = (state[14] << 16) | (state[14] >> 16)
            state[9] = state[9] + state[14]; state[4] ^= state[9]; state[4] = (state[4] << 12) | (state[4] >> 20)
            state[3] = state[3] + state[4]; state[14] ^= state[3]; state[14] = (state[14] << 8) | (state[14] >> 24)
            state[9] = state[9] + state[14]; state[4] ^= state[9]; state[4] = (state[4] << 7) | (state[4] >> 25)

        total += state[0]

    free(state)
    return total
