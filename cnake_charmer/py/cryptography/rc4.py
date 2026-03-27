"""RC4 stream cipher encryption of n bytes with fixed key.

Keywords: cryptography, rc4, stream cipher, encryption, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def rc4(n: int) -> tuple:
    """Encrypt n bytes using RC4 stream cipher with fixed key.

    Plaintext bytes are generated as (i * 37 + 13) % 256.
    Key is [0xDE, 0xAD, 0xBE, 0xEF, 0x42].

    Args:
        n: Number of bytes to encrypt.

    Returns:
        Tuple of (xor_checksum, byte_at_mid, last_byte).
    """
    # Key Schedule Algorithm (KSA)
    key = [0xDE, 0xAD, 0xBE, 0xEF, 0x42]
    key_len = 5
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % key_len]) % 256
        S[i], S[j] = S[j], S[i]

    # Pseudo-Random Generation Algorithm (PRGA) + encryption
    xor_checksum = 0
    byte_at_mid = 0
    last_byte = 0
    mid = n // 2
    ii = 0
    jj = 0
    for idx in range(n):
        ii = (ii + 1) % 256
        jj = (jj + S[ii]) % 256
        S[ii], S[jj] = S[jj], S[ii]
        k = S[(S[ii] + S[jj]) % 256]
        plaintext = (idx * 37 + 13) % 256
        encrypted = plaintext ^ k
        xor_checksum ^= encrypted
        last_byte = encrypted
        if idx == mid:
            byte_at_mid = encrypted

    return (xor_checksum, byte_at_mid, last_byte)
