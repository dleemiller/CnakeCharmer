# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
FNV-1a style rolling hash computation (Cython-optimized).

Keywords: cryptography, hash, fnv, rolling, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def simple_hash(int n):
    """Compute an FNV-1a style rolling hash over n bytes using typed arithmetic.

    Args:
        n: Number of bytes to hash.

    Returns:
        Final 64-bit hash value.
    """
    cdef int i
    cdef unsigned char b
    cdef unsigned long long h = 14695981039346656037ULL
    cdef unsigned long long FNV_PRIME = 1099511628211ULL

    for i in range(n):
        b = (i * 7 + 3) % 256
        h = h ^ b
        h = h * FNV_PRIME

    return h
