# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count matching pairs of rolling hashes among n strings (Cython-optimized).

Keywords: string processing, hashing, rolling hash, comparison, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def string_hash_compare(int n):
    """Count hash-matching pairs using C array hash table."""
    cdef long long BASE = 31
    cdef long long MOD = 1000000007
    cdef int i, j, ch
    cdef long long h
    cdef long long total = 0

    # Use a hash table via C array (open addressing)
    cdef int table_size = n * 4  # Load factor ~0.25
    cdef long long *keys = <long long *>malloc(table_size * sizeof(long long))
    cdef int *counts = <int *>malloc(table_size * sizeof(int))
    cdef char *used = <char *>malloc(table_size * sizeof(char))
    if not keys or not counts or not used:
        raise MemoryError()

    memset(used, 0, table_size * sizeof(char))

    cdef int idx

    for i in range(n):
        h = 0
        for j in range(8):
            ch = (i * j + 3) % 26
            h = (h * BASE + ch) % MOD

        # Insert into hash table
        idx = <int>(h % table_size)
        while used[idx]:
            if keys[idx] == h:
                counts[idx] += 1
                break
            idx += 1
            if idx >= table_size:
                idx = 0
        else:
            used[idx] = 1
            keys[idx] = h
            counts[idx] = 1

    # Count pairs
    for i in range(table_size):
        if used[i]:
            total += <long long>counts[i] * (counts[i] - 1) / 2

    free(keys)
    free(counts)
    free(used)
    return total
