# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""XTEA block cipher with final block and checksum return (Cython-optimized).

Keywords: cryptography, xtea, block cipher, feistel, encryption, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

DEF MASK32 = 0xFFFFFFFF
DEF DELTA = 0x9E3779B9


@cython_benchmark(syntax="cy", args=(40000,))
def xtea(int n):
    """Encrypt n blocks with XTEA and return final block plus checksum."""
    cdef unsigned long long v0, v1, s
    cdef unsigned long long key[4]
    cdef unsigned long long checksum = 0
    cdef unsigned long long final_v0 = 0, final_v1 = 0
    cdef int i, j

    key[0] = 0x01234567
    key[1] = 0x89ABCDEF
    key[2] = 0xFEDCBA98
    key[3] = 0x76543210

    for i in range(n):
        v0 = (<unsigned long long>i * 2654435761 + 7) & MASK32
        v1 = (<unsigned long long>i * 2246822519 + 13) & MASK32
        s = 0

        for j in range(64):
            v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + key[s & 3]))) & MASK32
            s = (s + DELTA) & MASK32
            v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + key[(s >> 11) & 3]))) & MASK32

        checksum = (checksum + v0 + v1) & MASK32
        final_v0 = v0
        final_v1 = v1

    return (int(final_v0), int(final_v1), int(checksum))
