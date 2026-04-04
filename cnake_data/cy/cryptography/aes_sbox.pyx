# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply AES S-box substitution to n bytes (Cython-optimized).

Keywords: cryptography, aes, sbox, galois, substitution, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def aes_sbox(int n):
    """Apply AES S-box substitution to n bytes and return checksum."""
    cdef int i, j, k
    cdef unsigned char inv[256]
    cdef unsigned char sbox[256]
    cdef unsigned char a, b, p, hi, s, rotated
    cdef long long checksum
    cdef unsigned char byte_val

    # Build GF(2^8) inverse table
    inv[0] = 0
    for i in range(1, 256):
        for j in range(1, 256):
            # GF(2^8) multiply i * j
            a = <unsigned char>i
            b = <unsigned char>j
            p = 0
            for k in range(8):
                if b & 1:
                    p ^= a
                hi = a & 0x80
                a = (a << 1) & 0xFF
                if hi:
                    a ^= 0x1B
                b >>= 1
            if p == 1:
                inv[i] = <unsigned char>j
                break

    # Build S-box using affine transform
    for i in range(256):
        b = inv[i]
        s = b
        for k in range(1, 5):
            rotated = ((b << k) | (b >> (8 - k))) & 0xFF
            s ^= rotated
        sbox[i] = (s ^ 0x63) & 0xFF

    # Apply S-box to data
    checksum = 0
    for i in range(n):
        byte_val = (i * 7 + 3) % 256
        checksum += sbox[byte_val]

    return checksum
