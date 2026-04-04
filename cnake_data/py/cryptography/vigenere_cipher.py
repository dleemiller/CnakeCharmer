"""Vigenere cipher encryption with checksum computation.

Keywords: vigenere, cipher, encryption, cryptography, polyalphabetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def vigenere_cipher(n: int) -> tuple:
    """Encrypt n bytes using the Vigenere cipher and return checksums.

    Generates a deterministic plaintext sequence and encrypts it with
    a fixed key using modular addition (mod 256 for byte-level operation).

    Args:
        n: Number of bytes to encrypt.

    Returns:
        Tuple of (xor checksum of ciphertext, sum of ciphertext mod 2^64,
        last ciphertext byte).
    """
    key = [
        0x4B,
        0x65,
        0x79,
        0x31,
        0x32,
        0x33,
        0x34,
        0x35,
        0x41,
        0x42,
        0x43,
        0x44,
        0x45,
        0x46,
        0x47,
        0x48,
    ]
    key_len = len(key)
    mask64 = 0xFFFFFFFFFFFFFFFF

    xor_check = 0
    total = 0
    last_byte = 0

    for i in range(n):
        # Deterministic plaintext: simple LCG-like sequence
        plain = ((i * 1103515245 + 12345) >> 8) & 0xFF
        cipher = (plain + key[i % key_len]) & 0xFF
        xor_check ^= cipher
        total = (total + cipher) & mask64
        last_byte = cipher

    return (xor_check, total, last_byte)
