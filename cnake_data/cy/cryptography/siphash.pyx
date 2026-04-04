# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SipHash-2-4 of n 8-byte blocks (Cython-optimized).

Keywords: siphash, hash, cryptography, bit operations, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def siphash(int n):
    """Compute SipHash-2-4 using typed 64-bit arithmetic."""
    cdef unsigned long long MASK64 = 0xFFFFFFFFFFFFFFFF
    cdef unsigned long long v0, v1, v2, v3, mi, b
    cdef unsigned long long k0, k1
    cdef int i
    cdef int total_len = n * 8

    k0 = 0x0706050403020100ULL
    k1 = 0x0F0E0D0C0B0A0908ULL

    v0 = k0 ^ 0x736F6D6570736575ULL
    v1 = k1 ^ 0x646F72616E646F6DULL
    v2 = k0 ^ 0x6C7967656E657261ULL
    v3 = k1 ^ 0x7465646279746573ULL

    for i in range(n):
        mi = <unsigned long long>(i * 7 + 3)

        v3 ^= mi
        # Round 1
        v0 = (v0 + v1) & MASK64
        v1 = ((v1 << 13) | (v1 >> 51)) & MASK64
        v1 ^= v0
        v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
        v2 = (v2 + v3) & MASK64
        v3 = ((v3 << 16) | (v3 >> 48)) & MASK64
        v3 ^= v2
        v0 = (v0 + v3) & MASK64
        v3 = ((v3 << 21) | (v3 >> 43)) & MASK64
        v3 ^= v0
        v2 = (v2 + v1) & MASK64
        v1 = ((v1 << 17) | (v1 >> 47)) & MASK64
        v1 ^= v2
        v2 = ((v2 << 32) | (v2 >> 32)) & MASK64
        # Round 2
        v0 = (v0 + v1) & MASK64
        v1 = ((v1 << 13) | (v1 >> 51)) & MASK64
        v1 ^= v0
        v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
        v2 = (v2 + v3) & MASK64
        v3 = ((v3 << 16) | (v3 >> 48)) & MASK64
        v3 ^= v2
        v0 = (v0 + v3) & MASK64
        v3 = ((v3 << 21) | (v3 >> 43)) & MASK64
        v3 ^= v0
        v2 = (v2 + v1) & MASK64
        v1 = ((v1 << 17) | (v1 >> 47)) & MASK64
        v1 ^= v2
        v2 = ((v2 << 32) | (v2 >> 32)) & MASK64

        v0 ^= mi

    # Finalization
    b = (<unsigned long long>(total_len & 0xFF)) << 56
    v3 ^= b
    # 2 rounds
    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64
    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64
    v0 ^= b

    v2 ^= 0xFF
    # 4 finalization rounds
    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64

    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64

    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64

    v0 = (v0 + v1) & MASK64; v1 = ((v1 << 13) | (v1 >> 51)) & MASK64; v1 ^= v0; v0 = ((v0 << 32) | (v0 >> 32)) & MASK64
    v2 = (v2 + v3) & MASK64; v3 = ((v3 << 16) | (v3 >> 48)) & MASK64; v3 ^= v2
    v0 = (v0 + v3) & MASK64; v3 = ((v3 << 21) | (v3 >> 43)) & MASK64; v3 ^= v0
    v2 = (v2 + v1) & MASK64; v1 = ((v1 << 17) | (v1 >> 47)) & MASK64; v1 ^= v2; v2 = ((v2 << 32) | (v2 >> 32)) & MASK64

    return (v0 ^ v1 ^ v2 ^ v3) & MASK64
