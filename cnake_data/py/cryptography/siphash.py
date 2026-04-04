"""SipHash-2-4 of n 8-byte blocks.

Keywords: siphash, hash, cryptography, bit operations, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def siphash(n: int) -> int:
    """Compute SipHash-2-4 over n 8-byte blocks.

    Each block is (i * 7 + 3) as a 64-bit little-endian value.
    Key is fixed: k0=0x0706050403020100, k1=0x0f0e0d0c0b0a0908.

    Args:
        n: Number of 8-byte blocks to hash.

    Returns:
        Final 64-bit hash value as int.
    """
    MASK64 = 0xFFFFFFFFFFFFFFFF

    def rotl(x, b):
        return ((x << b) | (x >> (64 - b))) & MASK64

    def sipround(v0, v1, v2, v3):
        v0 = (v0 + v1) & MASK64
        v1 = rotl(v1, 13)
        v1 ^= v0
        v0 = rotl(v0, 32)
        v2 = (v2 + v3) & MASK64
        v3 = rotl(v3, 16)
        v3 ^= v2
        v0 = (v0 + v3) & MASK64
        v3 = rotl(v3, 21)
        v3 ^= v0
        v2 = (v2 + v1) & MASK64
        v1 = rotl(v1, 17)
        v1 ^= v2
        v2 = rotl(v2, 32)
        return v0, v1, v2, v3

    k0 = 0x0706050403020100
    k1 = 0x0F0E0D0C0B0A0908

    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    total_len = n * 8

    for i in range(n):
        mi = (i * 7 + 3) & MASK64
        v3 ^= mi
        # 2 rounds
        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
        v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
        v0 ^= mi

    # Finalization: last byte is total_len mod 256
    b = (total_len & 0xFF) << 56
    v3 ^= b
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0 ^= b

    v2 ^= 0xFF
    # 4 finalization rounds
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)
    v0, v1, v2, v3 = sipround(v0, v1, v2, v3)

    return (v0 ^ v1 ^ v2 ^ v3) & MASK64
