"""
XOR cipher encryption with checksum computation.

Keywords: cryptography, xor, cipher, encryption, checksum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def xor_cipher_checksum(n: int) -> int:
    """XOR-encrypt n bytes with a repeating 16-byte key and return checksum.

    Plaintext: b[i] = (i*7+3) % 256
    Key: k[j] = (j*13+11) % 256 for j in range(16)
    Encrypted: e[i] = b[i] ^ k[i % 16]
    Checksum: sum of all encrypted bytes.

    Args:
        n: Number of bytes to encrypt.

    Returns:
        Sum of all encrypted bytes.
    """
    # Generate key
    key = [0] * 16
    for j in range(16):
        key[j] = (j * 13 + 11) % 256

    # Encrypt and compute checksum
    checksum = 0
    last_byte = 0
    for i in range(n):
        plaintext = (i * 7 + 3) % 256
        encrypted = plaintext ^ key[i % 16]
        checksum += encrypted
        last_byte = encrypted

    return (checksum, last_byte)
