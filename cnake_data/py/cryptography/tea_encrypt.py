"""TEA (Tiny Encryption Algorithm) on n 64-bit blocks.

Keywords: tea, encryption, cryptography, block cipher, feistel, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF
DELTA = 0x9E3779B9


@python_benchmark(args=(50000,))
def tea_encrypt(n: int) -> int:
    """Encrypt n blocks with TEA and return sum of encrypted values.

    Each block (v0, v1) = (i*7+3, i*13+7) masked to 32 bits.
    Key = (1, 2, 3, 4). 32 rounds per block.

    Args:
        n: Number of 64-bit blocks to encrypt.

    Returns:
        Sum of all encrypted v0 and v1 values, mod 2^64.
    """
    k0, k1, k2, k3 = 1, 2, 3, 4
    mask = MASK32
    delta = DELTA
    total = 0

    for i in range(n):
        v0 = (i * 7 + 3) & mask
        v1 = (i * 13 + 7) & mask
        s = 0

        for _ in range(32):
            s = (s + delta) & mask
            v0 = (v0 + (((v1 << 4) + k0) ^ (v1 + s) ^ ((v1 >> 5) + k1))) & mask
            v1 = (v1 + (((v0 << 4) + k2) ^ (v0 + s) ^ ((v0 >> 5) + k3))) & mask

        total += v0 + v1

    return total & 0xFFFFFFFFFFFFFFFF
