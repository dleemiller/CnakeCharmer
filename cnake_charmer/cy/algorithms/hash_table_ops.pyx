# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Linear-probing hash table insert and lookup operations (Cython-optimized).

Keywords: algorithms, hash table, linear probing, lookup, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def hash_table_ops(int n):
    """Insert keys and count query hits using C array hash table."""
    cdef int table_size = 2 * n
    cdef int EMPTY = -1
    cdef int *table = <int *>malloc(table_size * sizeof(int))

    if not table:
        raise MemoryError()

    cdef int i, key, query, slot, hits, occupied

    # Initialize table to EMPTY
    for i in range(table_size):
        table[i] = EMPTY

    # Insert keys
    for i in range(n):
        key = (i * 31 + 17) % 100000
        slot = key % table_size
        while table[slot] != EMPTY:
            if table[slot] == key:
                break
            slot = (slot + 1) % table_size
        table[slot] = key

    # Count query hits
    hits = 0
    for i in range(n):
        query = (i * 37 + 13) % 100000
        slot = query % table_size
        while table[slot] != EMPTY:
            if table[slot] == query:
                hits += 1
                break
            slot = (slot + 1) % table_size

    # Count occupied slots
    occupied = 0
    for i in range(table_size):
        if table[i] != EMPTY:
            occupied += 1

    free(table)
    return (hits, occupied)
