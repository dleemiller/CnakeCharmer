"""SHA-256 hash of n deterministic bytes using pure Python.

Keywords: cryptography, sha256, hash, digest, block cipher, benchmark
"""

from cnake_data.benchmarks import python_benchmark

_K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]

_MASK = 0xFFFFFFFF


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


@python_benchmark(args=(50000,))
def sha256_blocks(n: int) -> tuple:
    """Compute SHA-256 over n deterministic bytes (i % 251).

    Args:
        n: Number of input bytes.

    Returns:
        Tuple of (h0, h3, h7) — three independent 32-bit words of the hash state.
    """
    # Build padded message
    msg = bytearray(i % 251 for i in range(n))
    bit_len = n * 8
    msg += b"\x80"
    while len(msg) % 64 != 56:
        msg += b"\x00"
    msg += bit_len.to_bytes(8, "big")

    # Initial hash values (sqrt of first 8 primes)
    h0 = 0x6A09E667
    h1 = 0xBB67AE85
    h2 = 0x3C6EF372
    h3 = 0xA54FF53A
    h4 = 0x510E527F
    h5 = 0x9B05688C
    h6 = 0x1F83D9AB
    h7 = 0x5BE0CD19

    W = [0] * 64

    for blk in range(0, len(msg), 64):
        block = msg[blk : blk + 64]

        # Message schedule
        for j in range(16):
            W[j] = (
                (block[4 * j] << 24)
                | (block[4 * j + 1] << 16)
                | (block[4 * j + 2] << 8)
                | block[4 * j + 3]
            )
        for j in range(16, 64):
            s0 = _rotr32(W[j - 15], 7) ^ _rotr32(W[j - 15], 18) ^ (W[j - 15] >> 3)
            s1 = _rotr32(W[j - 2], 17) ^ _rotr32(W[j - 2], 19) ^ (W[j - 2] >> 10)
            W[j] = (W[j - 16] + s0 + W[j - 7] + s1) & _MASK

        # Compression
        a, b, c, d, e, f, g, hh = h0, h1, h2, h3, h4, h5, h6, h7

        for j in range(64):
            S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (hh + S1 + ch + _K[j] + W[j]) & _MASK
            S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & _MASK
            hh = g
            g = f
            f = e
            e = (d + temp1) & _MASK
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & _MASK

        h0 = (h0 + a) & _MASK
        h1 = (h1 + b) & _MASK
        h2 = (h2 + c) & _MASK
        h3 = (h3 + d) & _MASK
        h4 = (h4 + e) & _MASK
        h5 = (h5 + f) & _MASK
        h6 = (h6 + g) & _MASK
        h7 = (h7 + hh) & _MASK

    return (h0, h3, h7)
