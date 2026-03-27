"""XTEA block cipher with final block and checksum return.

Keywords: cryptography, xtea, block cipher, feistel, encryption, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MASK32 = 0xFFFFFFFF
DELTA = 0x9E3779B9


@python_benchmark(args=(40000,))
def xtea(n: int) -> tuple:
    """Encrypt n blocks with XTEA and return final block plus checksum.

    Each block (v0, v1) is generated from block index.
    Key = (0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210).
    64 rounds per block.

    Args:
        n: Number of 64-bit blocks to encrypt.

    Returns:
        Tuple of (final_block_v0, final_block_v1, checksum).
    """
    k0 = 0x01234567
    k1 = 0x89ABCDEF
    k2 = 0xFEDCBA98
    k3 = 0x76543210
    key = [k0, k1, k2, k3]
    mask = MASK32
    delta = DELTA

    checksum = 0
    final_v0 = 0
    final_v1 = 0

    for i in range(n):
        v0 = (i * 2654435761 + 7) & mask
        v1 = (i * 2246822519 + 13) & mask
        s = 0

        for _ in range(64):
            v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + key[s & 3]))) & mask
            s = (s + delta) & mask
            v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + key[(s >> 11) & 3]))) & mask

        checksum = (checksum + v0 + v1) & MASK32
        final_v0 = v0
        final_v1 = v1

    return (final_v0, final_v1, checksum)
