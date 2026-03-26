# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
XOR cipher encryption with checksum computation (Cython-optimized).

Keywords: cryptography, xor, cipher, encryption, checksum, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def xor_cipher_checksum(int n):
    """XOR-encrypt n bytes using unsigned char C arrays and return checksum.

    Args:
        n: Number of bytes to encrypt.

    Returns:
        Sum of all encrypted bytes.
    """
    cdef int i, j
    cdef unsigned char plaintext, encrypted
    cdef long long checksum = 0

    cdef unsigned char *key = <unsigned char *>malloc(16 * sizeof(unsigned char))
    if key == NULL:
        raise MemoryError("Failed to allocate key array")

    # Generate key
    for j in range(16):
        key[j] = (j * 13 + 11) % 256

    # Encrypt and compute checksum
    for i in range(n):
        plaintext = (i * 7 + 3) % 256
        encrypted = plaintext ^ key[i % 16]
        checksum += encrypted

    free(key)
    return checksum
