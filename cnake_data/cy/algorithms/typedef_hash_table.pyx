# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Hash table with ctypedef'd key type, count collisions (Cython-optimized).

Keywords: algorithms, hash table, ctypedef, collision, cython, benchmark
"""

from libc.stdlib cimport free, calloc
from cnake_data.benchmarks import cython_benchmark

ctypedef unsigned long long uint64
ctypedef int[2] pair_t


@cython_benchmark(syntax="cy", args=(50000,))
def typedef_hash_table(int n):
    """Insert n values into a hash table and count collisions using ctypedef types."""
    cdef int table_size = n * 2
    cdef char *occupied = <char *>calloc(table_size, sizeof(char))
    if not occupied:
        raise MemoryError()

    cdef int collisions = 0
    cdef int i
    cdef uint64 key
    cdef int bucket

    for i in range(n):
        key = <uint64>i * <uint64>2654435761
        key = key & <uint64>0xFFFFFFFF
        bucket = <int>(key % <uint64>table_size)
        if occupied[bucket]:
            collisions += 1
        occupied[bucket] = 1

    free(occupied)
    return collisions
