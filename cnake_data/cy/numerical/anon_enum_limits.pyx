# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Hash table simulation using anonymous enum constants.

Keywords: numerical, enum, hash table, constants, simulation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark

cdef enum:
    MAX_SIZE = 1024
    BUCKET_COUNT = 256
    HASH_SEED = 0x9E3779B9


@cython_benchmark(syntax="cy", args=(100000,))
def anon_enum_limits(int n):
    """Simulate a hash table with enum-defined constants."""
    cdef unsigned int *table
    cdef char *occupied
    cdef int collisions = 0
    cdef int i, step, slot, placed, j
    cdef unsigned int val, bucket, region_start
    cdef unsigned int checksum
    cdef int region_size = MAX_SIZE // BUCKET_COUNT

    table = <unsigned int *>malloc(
        MAX_SIZE * sizeof(unsigned int)
    )
    occupied = <char *>malloc(MAX_SIZE * sizeof(char))
    if not table or not occupied:
        raise MemoryError()
    memset(table, 0, MAX_SIZE * sizeof(unsigned int))
    memset(occupied, 0, MAX_SIZE * sizeof(char))

    for i in range(n):
        val = (
            (<unsigned int>(<long long>i
             * <long long>HASH_SEED)
             ^ (<unsigned int>i >> 5))
            & <unsigned int>0xFFFFFFFF
        )
        bucket = val % BUCKET_COUNT
        region_start = (
            (bucket * region_size) % MAX_SIZE
        )

        placed = 0
        for step in range(region_size):
            slot = (region_start + step) % MAX_SIZE
            if occupied[slot] == 0:
                table[slot] = val
                occupied[slot] = 1
                placed = 1
                break
            else:
                collisions += 1

        if placed == 0:
            for slot in range(MAX_SIZE):
                if occupied[slot] == 0:
                    table[slot] = val
                    occupied[slot] = 1
                    break
                else:
                    collisions += 1

        if (i % (MAX_SIZE // 2) == 0) and (i > 0):
            for j in range(MAX_SIZE):
                if j % 4 == 0:
                    occupied[j] = 0
                    table[j] = 0

    checksum = 0
    for j in range(MAX_SIZE):
        if occupied[j]:
            checksum ^= table[j]

    free(table)
    free(occupied)
    return collisions + <int>(checksum & 0xFFFF)
