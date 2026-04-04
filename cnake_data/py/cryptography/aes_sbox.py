"""Apply AES S-box substitution to n bytes.

Keywords: cryptography, aes, sbox, galois, substitution, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def aes_sbox(n: int) -> int:
    """Apply AES S-box substitution to n bytes and return checksum.

    Builds the S-box from GF(2^8) multiplicative inverse + affine transform.
    Data: b[i] = (i*7+3) % 256. Returns sum of all substituted bytes.

    Args:
        n: Number of bytes to substitute.

    Returns:
        Checksum (sum of substituted bytes).
    """
    # Build GF(2^8) inverse table
    inv = [0] * 256

    # Use extended Euclidean / log-antilog via multiplication
    # Simpler: brute-force inverse in GF(2^8) with irreducible poly 0x11B
    def gf_mul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p

    for i in range(1, 256):
        for j in range(1, 256):
            if gf_mul(i, j) == 1:
                inv[i] = j
                break

    # Build S-box using affine transform
    sbox = [0] * 256
    for i in range(256):
        b = inv[i]
        # Affine transform: rotate and XOR with 0x63
        s = b
        for k in range(1, 5):
            s ^= ((b << k) | (b >> (8 - k))) & 0xFF
        sbox[i] = (s ^ 0x63) & 0xFF

    # Apply S-box to data
    checksum = 0
    for i in range(n):
        byte = (i * 7 + 3) % 256
        checksum += sbox[byte]

    return checksum
