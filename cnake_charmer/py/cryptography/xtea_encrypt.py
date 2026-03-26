"""XTEA block cipher encryption.

Keywords: xtea, encryption, cryptography, block cipher, feistel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF
DELTA = 0x9E3779B9


@python_benchmark(args=(50000,))
def xtea_encrypt(n: int) -> tuple:
    """Encrypt n blocks with XTEA and return checksums.

    Each block (v0, v1) is generated deterministically from the block index.
    Key = (0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0).
    32 rounds per block.

    Args:
        n: Number of 64-bit blocks to encrypt.

    Returns:
        Tuple of (xor of all encrypted v0, xor of all encrypted v1,
        sum of all encrypted values mod 2^64).
    """
    k0 = 0xDEADBEEF
    k1 = 0xCAFEBABE
    k2 = 0x12345678
    k3 = 0x9ABCDEF0
    key = [k0, k1, k2, k3]
    mask = MASK32
    delta = DELTA
    mask64 = 0xFFFFFFFFFFFFFFFF

    xor_v0 = 0
    xor_v1 = 0
    total = 0

    for i in range(n):
        v0 = (i * 2654435761) & mask
        v1 = (i * 2246822519 + 1) & mask
        s = 0

        for _ in range(32):
            v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + key[s & 3]))) & mask
            s = (s + delta) & mask
            v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + key[(s >> 11) & 3]))) & mask

        xor_v0 ^= v0
        xor_v1 ^= v1
        total = (total + v0 + v1) & mask64

    return (xor_v0, xor_v1, total)
