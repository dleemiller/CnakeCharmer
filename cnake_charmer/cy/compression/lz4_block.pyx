# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LZ4-style block compression of deterministic data (Cython-optimized).

Keywords: compression, lz4, hash table, block compression, sliding window, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef int HASH_BITS  = 12
cdef int HASH_SIZE  = 1 << 12   # 4096
cdef unsigned int HASH_MULT = 2654435761
cdef int MIN_MATCH  = 4
cdef int MAX_OFFSET = 65535
cdef int MAX_MATCH  = 255 + 4   # 259


@cython_benchmark(syntax="cy", args=(200000,))
def lz4_block(int n):
    """Compress n bytes using LZ4-style block encoding with C arrays.

    Returns:
        Tuple of (literal_count, match_count, total_tokens).
    """
    cdef unsigned char *data = <unsigned char *>malloc(n * sizeof(unsigned char))
    # Stack-allocated hash table (4096 ints = 16 KB)
    cdef int hash_table[4096]
    if data == NULL:
        raise MemoryError()

    cdef int i, pos, candidate, match_len, max_len, h
    cdef unsigned int val
    cdef int literal_count = 0
    cdef int match_count = 0
    cdef int total_match_len = 0

    with nogil:
        # Generate source data: letters with period 26
        for i in range(n):
            data[i] = <unsigned char>(((i * 7 + 3) % 26 + 65))

        # Initialise hash table to "empty"
        for i in range(HASH_SIZE):
            hash_table[i] = -1

        pos = 0
        while pos <= n - MIN_MATCH:
            # 4-byte little-endian word at pos
            val = (
                <unsigned int>data[pos]
                | (<unsigned int>data[pos + 1] << 8)
                | (<unsigned int>data[pos + 2] << 16)
                | (<unsigned int>data[pos + 3] << 24)
            )
            h = <int>((val * HASH_MULT) >> (32 - HASH_BITS))

            candidate = hash_table[h]
            hash_table[h] = pos

            if candidate >= 0 and 0 < pos - candidate <= MAX_OFFSET:
                # Measure match length
                match_len = 0
                max_len = MAX_MATCH
                if n - pos < max_len:
                    max_len = n - pos
                if pos - candidate < max_len:
                    max_len = pos - candidate
                while match_len < max_len and data[candidate + match_len] == data[pos + match_len]:
                    match_len += 1

                if match_len >= MIN_MATCH:
                    match_count += 1
                    total_match_len += match_len
                    pos += match_len
                    continue

            literal_count += 1
            pos += 1

        literal_count += n - pos

    free(data)
    return (literal_count, match_count, total_match_len)
