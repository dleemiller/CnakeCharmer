# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Move-to-front encoding transform with checksums (Cython-optimized).

Keywords: compression, move-to-front, MTF, transform, encoding, cython, benchmark
"""


from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def mtf_encode(int n):
    """Apply move-to-front encoding to n deterministic bytes."""
    cdef int i, j, byte, pos, val
    cdef long long encoded_checksum = 0
    cdef long long alpha_checksum = 0
    cdef long long MOD = (1LL << 63)
    cdef int alphabet[256]

    # Initialize alphabet
    for i in range(256):
        alphabet[i] = i

    for i in range(n):
        byte = (i * 13 + 7) % 256

        # Find position of byte in alphabet
        pos = 0
        for j in range(256):
            if alphabet[j] == byte:
                pos = j
                break

        # Add to encoded checksum
        encoded_checksum = (encoded_checksum + <long long>pos * <long long>(i + 1)) % MOD

        # Move to front
        if pos > 0:
            val = alphabet[pos]
            for j in range(pos, 0, -1):
                alphabet[j] = alphabet[j - 1]
            alphabet[0] = val

    # Compute alphabet state checksum
    for i in range(256):
        alpha_checksum = (alpha_checksum + <long long>alphabet[i] * <long long>(i + 1)) % MOD

    return (encoded_checksum, alpha_checksum)
