# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute hash-based checksum from rapidly created Pair objects.

Keywords: algorithms, pair, freelist, extension type, hashing, cython, benchmark
"""

cimport cython
from cnake_data.benchmarks import cython_benchmark


@cython.freelist(128)
cdef class Pair:
    """Simple key-value pair with freelist optimization."""
    cdef unsigned int key
    cdef unsigned int value

    def __cinit__(self, unsigned int key, unsigned int value):
        self.key = key
        self.value = value

    cdef unsigned int hash_code(self):
        return (
            (self.key * <unsigned int>2654435761)
            ^ (self.value * <unsigned int>1664525)
        ) & <unsigned int>0xFFFFFFFF


@cython_benchmark(syntax="cy", args=(100000,))
def freelist_pair_hash(int n):
    """Create n pairs, compute cumulative hash checksum."""
    cdef unsigned int checksum = 0
    cdef int i
    cdef unsigned int key, value
    cdef Pair p

    for i in range(n):
        key = (
            <unsigned int>i * <unsigned int>1103515245
            + <unsigned int>12345
        )
        value = (
            <unsigned int>i * <unsigned int>214013
            + <unsigned int>2531011
        )
        p = Pair(key, value)
        checksum ^= p.hash_code()
    return <long long>checksum
