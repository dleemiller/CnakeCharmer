# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Caesar cipher encrypt and decrypt with round-trip verification (Cython-optimized).

Keywords: cryptography, caesar, cipher, encryption, decryption, checksum, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def caesar_cipher(int n):
    """Caesar-cipher encrypt n bytes, verify round-trip, return checksum.

    Args:
        n: Number of bytes to process.

    Returns:
        Sum of all encrypted bytes.
    """
    cdef int i
    cdef unsigned char plain, encrypted, decrypted
    cdef long long checksum = 0

    cdef unsigned char *buf = <unsigned char *>malloc(n * sizeof(unsigned char))
    if buf == NULL:
        raise MemoryError("Failed to allocate buffer")

    # Encrypt and verify round-trip
    for i in range(n):
        plain = (i * 7 + 3) % 256
        encrypted = (plain + 13) % 256
        decrypted = (encrypted - 13) % 256
        if decrypted != plain:
            free(buf)
            raise ValueError("Round-trip failed")
        buf[i] = encrypted

    # Compute checksum
    for i in range(n):
        checksum += buf[i]

    free(buf)
    return checksum
