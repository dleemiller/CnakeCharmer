"""Move-to-front encoding transform with checksums.

Keywords: compression, move-to-front, MTF, transform, encoding, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def mtf_encode(n: int) -> tuple:
    """Apply move-to-front encoding to n deterministic bytes.

    Input bytes: b[i] = (i * 13 + 7) % 256.
    Returns (encoded checksum, final alphabet state checksum) where
    checksum = sum of encoded[i] * (i + 1) mod 2^63.

    Args:
        n: Number of bytes to encode.

    Returns:
        Tuple of (encoded checksum, alphabet state checksum).
    """
    MOD = 1 << 63

    # Initialize alphabet as list [0, 1, 2, ..., 255]
    alphabet = list(range(256))

    encoded_checksum = 0

    for i in range(n):
        byte = (i * 13 + 7) % 256

        # Find position of byte in alphabet
        pos = 0
        for j in range(256):
            if alphabet[j] == byte:
                pos = j
                break

        # Add to encoded checksum
        encoded_checksum = (encoded_checksum + pos * (i + 1)) % MOD

        # Move to front
        if pos > 0:
            val = alphabet[pos]
            for j in range(pos, 0, -1):
                alphabet[j] = alphabet[j - 1]
            alphabet[0] = val

    # Compute alphabet state checksum
    alpha_checksum = 0
    for i in range(256):
        alpha_checksum = (alpha_checksum + alphabet[i] * (i + 1)) % MOD

    return (encoded_checksum, alpha_checksum)
