# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build signed-variable hash mappings and aggregate clause hashes (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef void _variable_mapping_impl(
    unsigned int *pos,
    unsigned int *neg,
    int n_vars,
    unsigned int seed,
    int draws,
    int clause_len,
    unsigned int *checksum_out,
    int *collisions_out,
    unsigned int *last_hash_out,
) noexcept nogil:
    cdef unsigned int x = seed
    cdef int v, i, j, idx
    cdef unsigned int checksum = 0
    cdef int collisions = 0
    cdef unsigned int last_hash = 0
    cdef unsigned int h

    for v in range(1, n_vars + 1):
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        pos[v] = x
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        neg[v] = x

    for i in range(draws):
        h = 0
        for j in range(clause_len):
            idx = ((i * 1103515245 + j * 12345 + seed) & MASK32) % n_vars + 1
            if ((i + j) & 1) == 0:
                h += pos[idx]
            else:
                h += neg[idx]
        if (h & 1023) == (last_hash & 1023):
            collisions += 1
        checksum = (checksum + (h & MASK32)) & MASK32
        last_hash = h

    checksum_out[0] = checksum
    collisions_out[0] = collisions
    last_hash_out[0] = last_hash


@cython_benchmark(syntax="cy", args=(320, 1337, 240000, 5))
def variable_mapping_checksum(int n_vars, unsigned int seed, int draws, int clause_len):
    cdef unsigned int *pos = <unsigned int *>malloc((n_vars + 1) * sizeof(unsigned int))
    cdef unsigned int *neg = <unsigned int *>malloc((n_vars + 1) * sizeof(unsigned int))
    cdef unsigned int checksum = 0
    cdef int collisions = 0
    cdef unsigned int last_hash = 0

    if not pos or not neg:
        free(pos)
        free(neg)
        raise MemoryError()

    with nogil:
        _variable_mapping_impl(
            pos,
            neg,
            n_vars,
            seed,
            draws,
            clause_len,
            &checksum,
            &collisions,
            &last_hash,
        )

    free(pos)
    free(neg)
    return (checksum, collisions, last_hash & MASK32)
