# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Vigenere cipher encryption with checksum computation (Cython-optimized).

Keywords: vigenere, cipher, encryption, cryptography, polyalphabetic, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def vigenere_cipher(int n):
    """Encrypt n bytes using the Vigenere cipher and return checksums."""
    cdef int i
    cdef unsigned int plain, cipher
    cdef int xor_check = 0
    cdef unsigned long long total = 0
    cdef int last_byte = 0
    cdef int key_len = 16

    cdef int key[16]
    key[0] = 0x4B; key[1] = 0x65; key[2] = 0x79; key[3] = 0x31
    key[4] = 0x32; key[5] = 0x33; key[6] = 0x34; key[7] = 0x35
    key[8] = 0x41; key[9] = 0x42; key[10] = 0x43; key[11] = 0x44
    key[12] = 0x45; key[13] = 0x46; key[14] = 0x47; key[15] = 0x48

    for i in range(n):
        plain = ((<unsigned int>i * 1103515245 + 12345) >> 8) & 0xFF
        cipher = (plain + <unsigned int>key[i % key_len]) & 0xFF
        xor_check = xor_check ^ cipher
        total = (total + cipher) & 0xFFFFFFFFFFFFFFFF
        last_byte = cipher

    return (xor_check, int(total), last_byte)
