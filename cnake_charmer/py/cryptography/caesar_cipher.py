"""
Caesar cipher encrypt and decrypt with round-trip verification.

Keywords: cryptography, caesar, cipher, encryption, decryption, checksum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def caesar_cipher(n: int) -> int:
    """Caesar-cipher encrypt n bytes with shift=13, decrypt, verify round-trip,
    and return the checksum (sum) of the encrypted bytes.

    Plaintext: b[i] = (i*7+3) % 256
    Encrypt: e[i] = (b[i] + 13) % 256
    Decrypt: d[i] = (e[i] - 13) % 256
    Verify: d[i] == b[i] for all i (else raise)
    Return: sum of all encrypted bytes.

    Args:
        n: Number of bytes to process.

    Returns:
        Sum of all encrypted bytes.
    """
    checksum = 0
    for i in range(n):
        plain = (i * 7 + 3) % 256
        encrypted = (plain + 13) % 256
        decrypted = (encrypted - 13) % 256
        if decrypted != plain:
            raise ValueError("Round-trip failed")
        checksum += encrypted

    return checksum
