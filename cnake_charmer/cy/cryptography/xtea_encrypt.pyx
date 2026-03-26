# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""XTEA block cipher encryption (Cython-optimized).

Keywords: xtea, encryption, cryptography, block cipher, feistel, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

DEF DELTA = 0x9E3779B9
DEF MASK32 = 0xFFFFFFFF


@cython_benchmark(syntax="cy", args=(50000,))
def xtea_encrypt(int n):
    """Encrypt n blocks with XTEA and return checksums."""
    cdef unsigned long long v0, v1, s
    cdef unsigned long long key[4]
    cdef unsigned long long xor_v0 = 0, xor_v1 = 0
    cdef unsigned long long total = 0
    cdef int i, j

    key[0] = 0xDEADBEEF
    key[1] = 0xCAFEBABE
    key[2] = 0x12345678
    key[3] = 0x9ABCDEF0

    for i in range(n):
        v0 = (<unsigned long long>i * 2654435761) & MASK32
        v1 = (<unsigned long long>i * 2246822519 + 1) & MASK32
        s = 0

        for j in range(32):
            v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + key[s & 3]))) & MASK32
            s = (s + DELTA) & MASK32
            v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + key[(s >> 11) & 3]))) & MASK32

        xor_v0 = xor_v0 ^ v0
        xor_v1 = xor_v1 ^ v1
        total = (total + v0 + v1) & 0xFFFFFFFFFFFFFFFF

    return (int(xor_v0), int(xor_v1), int(total))
