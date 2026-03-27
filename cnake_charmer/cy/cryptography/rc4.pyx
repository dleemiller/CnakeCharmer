# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""RC4 stream cipher encryption of n bytes with fixed key (Cython-optimized).

Keywords: cryptography, rc4, stream cipher, encryption, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def rc4(int n):
    """Encrypt n bytes using RC4 stream cipher with fixed key."""
    cdef unsigned char S[256]
    cdef unsigned char key[5]
    cdef int ii, jj, idx
    cdef unsigned char temp, k_byte
    cdef int xor_checksum = 0
    cdef int byte_at_mid = 0
    cdef int last_byte = 0
    cdef int mid = n // 2
    cdef int plaintext, encrypted

    key[0] = 0xDE
    key[1] = 0xAD
    key[2] = 0xBE
    key[3] = 0xEF
    key[4] = 0x42

    # KSA
    for ii in range(256):
        S[ii] = ii

    jj = 0
    for ii in range(256):
        jj = (jj + S[ii] + key[ii % 5]) % 256
        temp = S[ii]
        S[ii] = S[jj]
        S[jj] = temp

    # PRGA + encryption
    ii = 0
    jj = 0
    for idx in range(n):
        ii = (ii + 1) % 256
        jj = (jj + S[ii]) % 256
        temp = S[ii]
        S[ii] = S[jj]
        S[jj] = temp
        k_byte = S[(S[ii] + S[jj]) % 256]
        plaintext = (idx * 37 + 13) % 256
        encrypted = plaintext ^ k_byte
        xor_checksum = xor_checksum ^ encrypted
        last_byte = encrypted
        if idx == mid:
            byte_at_mid = encrypted

    return (xor_checksum, byte_at_mid, last_byte)
