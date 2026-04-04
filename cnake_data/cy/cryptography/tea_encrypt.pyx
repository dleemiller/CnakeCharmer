# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""TEA (Tiny Encryption Algorithm) on n 64-bit blocks (Cython-optimized).

Keywords: tea, encryption, cryptography, block cipher, feistel, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

DEF DELTA = 0x9E3779B9
DEF MASK32 = 0xFFFFFFFF


@cython_benchmark(syntax="cy", args=(50000,))
def tea_encrypt(int n):
    """Encrypt n blocks with TEA and return sum of encrypted values."""
    cdef unsigned long long v0, v1, s, a, b, c
    cdef unsigned long long k0, k1, k2, k3
    cdef unsigned long long total
    cdef int i, j

    k0 = 1
    k1 = 2
    k2 = 3
    k3 = 4
    total = 0

    for i in range(n):
        v0 = (<unsigned long long>i * 7 + 3) & MASK32
        v1 = (<unsigned long long>i * 13 + 7) & MASK32
        s = 0

        for j in range(32):
            s = (s + DELTA) & MASK32
            a = ((v1 << 4) + k0)
            b = (v1 + s)
            c = ((v1 >> 5) + k1)
            v0 = (v0 + (a ^ b ^ c)) & MASK32
            a = ((v0 << 4) + k2)
            b = (v0 + s)
            c = ((v0 >> 5) + k3)
            v1 = (v1 + (a ^ b ^ c)) & MASK32

        total += v0 + v1

    return int(total & 0xFFFFFFFFFFFFFFFF)
