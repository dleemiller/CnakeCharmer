# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
XOR chain cipher with CBC-like chaining within 8-byte blocks (Cython-optimized).

Keywords: cryptography, xor, chain, cipher, cbc, chaining, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_data.benchmarks import cython_benchmark


N = 10000


@cython_benchmark(syntax="cy", args=(N,))
def xor_chain_cipher(int n):
    """Apply XOR chain cipher on n blocks of 8 bytes with 16-round key schedule.

    Args:
        n: Number of 8-byte blocks to process.

    Returns:
        Tuple of (total_checksum, first_block_byte0, last_block_byte7).
    """
    cdef int i, j, k, key
    cdef long long total_checksum = 0
    cdef unsigned char first_block_byte0 = 0
    cdef unsigned char last_block_byte7 = 0

    cdef unsigned char *data = <unsigned char *>malloc(8 * sizeof(unsigned char))
    cdef unsigned char *cipher = <unsigned char *>malloc(8 * sizeof(unsigned char))
    cdef int *keys = <int *>malloc(16 * sizeof(int))

    if data == NULL or cipher == NULL or keys == NULL:
        if data != NULL:
            free(data)
        if cipher != NULL:
            free(cipher)
        if keys != NULL:
            free(keys)
        raise MemoryError("Failed to allocate arrays")

    # Key schedule
    for i in range(16):
        keys[i] = (i * 37 + 11) % 8

    with nogil:
        for i in range(n):
            # Generate block deterministically
            for j in range(8):
                data[j] = (i * 7 + j * 13 + 42) % 256

            # Apply 16 rounds of chained XOR
            for k in range(16):
                key = keys[k]
                cipher[0] = data[key % 8]
                for j in range(1, 8):
                    cipher[j] = data[(key + j) % 8] ^ cipher[j - 1]
                # Copy cipher back to data
                memcpy(data, cipher, 8)

            # Accumulate checksum
            for j in range(8):
                total_checksum += data[j]

            if i == 0:
                first_block_byte0 = data[0]
            if i == n - 1:
                last_block_byte7 = data[7]

    free(data)
    free(cipher)
    free(keys)

    return (total_checksum, <int>first_block_byte0, <int>last_block_byte7)
