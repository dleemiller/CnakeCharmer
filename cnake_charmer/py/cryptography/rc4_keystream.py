"""
RC4 keystream generation and byte sum.

Keywords: cryptography, rc4, keystream, stream cipher, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def rc4_keystream(n: int) -> int:
    """Generate n bytes of RC4 keystream with key=[1,2,3,4,5] and return sum.

    Implements the standard RC4 KSA and PRGA algorithms.

    Args:
        n: Number of keystream bytes to generate.

    Returns:
        Sum of all generated keystream bytes.
    """
    # Key Schedule Algorithm (KSA)
    key = [1, 2, 3, 4, 5]
    key_len = 5
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % key_len]) % 256
        S[i], S[j] = S[j], S[i]

    # Pseudo-Random Generation Algorithm (PRGA)
    total = 0
    last_byte = 0
    i = 0
    j = 0
    for _ in range(n):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        k = S[(S[i] + S[j]) % 256]
        total += k
        last_byte = k

    return (total, last_byte)
