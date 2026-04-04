# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SHA-256 hash of n deterministic bytes (Cython-optimized).

Keywords: cryptography, sha256, hash, digest, block cipher, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

# SHA-256 round constants (cube roots of first 64 primes)
_SHA256_K = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
)


cdef inline unsigned int rotr32(unsigned int x, int n) noexcept nogil:
    return (x >> n) | (x << (32 - n))


@cython_benchmark(syntax="cy", args=(50000,))
def sha256_blocks(int n):
    """Compute SHA-256 over n deterministic bytes (i % 251).

    Returns:
        Tuple of (h0, h3, h7).
    """
    cdef int padded_len = ((n + 72) // 64) * 64  # = ceil((n+9)/64)*64
    cdef unsigned char *msg = <unsigned char *>malloc(padded_len)
    if msg == NULL:
        raise MemoryError()

    # Fill data + padding
    cdef int i, j, blk
    for i in range(n):
        msg[i] = i % 251
    msg[n] = 0x80
    for i in range(n + 1, padded_len - 8):
        msg[i] = 0
    # Big-endian bit length
    cdef unsigned long long bit_len = <unsigned long long>n * 8
    msg[padded_len - 8] = <unsigned char>(bit_len >> 56)
    msg[padded_len - 7] = <unsigned char>((bit_len >> 48) & 0xFF)
    msg[padded_len - 6] = <unsigned char>((bit_len >> 40) & 0xFF)
    msg[padded_len - 5] = <unsigned char>((bit_len >> 32) & 0xFF)
    msg[padded_len - 4] = <unsigned char>((bit_len >> 24) & 0xFF)
    msg[padded_len - 3] = <unsigned char>((bit_len >> 16) & 0xFF)
    msg[padded_len - 2] = <unsigned char>((bit_len >> 8) & 0xFF)
    msg[padded_len - 1] = <unsigned char>(bit_len & 0xFF)

    # Load K constants into C array
    cdef unsigned int K[64]
    for i in range(64):
        K[i] = _SHA256_K[i]

    # Initial hash state
    cdef unsigned int h0 = 0x6A09E667
    cdef unsigned int h1 = 0xBB67AE85
    cdef unsigned int h2 = 0x3C6EF372
    cdef unsigned int h3 = 0xA54FF53A
    cdef unsigned int h4 = 0x510E527F
    cdef unsigned int h5 = 0x9B05688C
    cdef unsigned int h6 = 0x1F83D9AB
    cdef unsigned int h7 = 0x5BE0CD19

    cdef unsigned int W[64]
    cdef unsigned int a, b, c, d, e, f, g, hh
    cdef unsigned int s0, s1, S0, S1, ch, maj, temp1, temp2
    cdef unsigned char *block

    with nogil:
        for blk in range(0, padded_len, 64):
            block = msg + blk

            # Message schedule
            for j in range(16):
                W[j] = (
                    (<unsigned int>block[4 * j] << 24)
                    | (<unsigned int>block[4 * j + 1] << 16)
                    | (<unsigned int>block[4 * j + 2] << 8)
                    | <unsigned int>block[4 * j + 3]
                )
            for j in range(16, 64):
                s0 = rotr32(W[j - 15], 7) ^ rotr32(W[j - 15], 18) ^ (W[j - 15] >> 3)
                s1 = rotr32(W[j - 2], 17) ^ rotr32(W[j - 2], 19) ^ (W[j - 2] >> 10)
                W[j] = W[j - 16] + s0 + W[j - 7] + s1

            # Compression
            a = h0; b = h1; c = h2; d = h3
            e = h4; f = h5; g = h6; hh = h7

            for j in range(64):
                S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)
                ch = (e & f) ^ (~e & g)
                temp1 = hh + S1 + ch + K[j] + W[j]
                S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)
                maj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = S0 + maj
                hh = g; g = f; f = e
                e = d + temp1
                d = c; c = b; b = a
                a = temp1 + temp2

            h0 += a; h1 += b; h2 += c; h3 += d
            h4 += e; h5 += f; h6 += g; h7 += hh

    free(msg)
    return (h0, h3, h7)
