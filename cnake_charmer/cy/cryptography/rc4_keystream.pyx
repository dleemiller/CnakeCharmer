# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
RC4 keystream generation and byte sum (Cython-optimized).

Keywords: cryptography, rc4, keystream, stream cipher, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def rc4_keystream(int n):
    """Generate n bytes of RC4 keystream with key=[1,2,3,4,5] and return sum.

    Uses a C unsigned char[256] array for the state.

    Args:
        n: Number of keystream bytes to generate.

    Returns:
        Sum of all generated keystream bytes.
    """
    cdef unsigned char S[256]
    cdef unsigned char key[5]
    cdef int ii, jj, kk
    cdef unsigned char temp
    cdef long long total = 0

    key[0] = 1
    key[1] = 2
    key[2] = 3
    key[3] = 4
    key[4] = 5

    # Key Schedule Algorithm (KSA)
    for ii in range(256):
        S[ii] = ii

    jj = 0
    for ii in range(256):
        jj = (jj + S[ii] + key[ii % 5]) % 256
        temp = S[ii]
        S[ii] = S[jj]
        S[jj] = temp

    # Pseudo-Random Generation Algorithm (PRGA)
    ii = 0
    jj = 0
    for kk in range(n):
        ii = (ii + 1) % 256
        jj = (jj + S[ii]) % 256
        temp = S[ii]
        S[ii] = S[jj]
        S[jj] = temp
        total += S[(S[ii] + S[jj]) % 256]

    return total
