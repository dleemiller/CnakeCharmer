# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count overlapping interval pairs using parallel C arrays and in-place sweep (Cython).

Keywords: algorithms, intervals, overlap, comparison, hash, cdef class, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_data.benchmarks import cython_benchmark


# ---------------------------------------------------------------------------
# C-level sort helpers
# ---------------------------------------------------------------------------

# Packed struct-like approach: interleave starts/ends in a single array of
# pairs so qsort can swap them atomically.  Layout: [s0, e0, s1, e1, ...]
cdef int _cmp_pairs(const void *a, const void *b) noexcept nogil:
    """Compare two (start, end) pairs by start first, then end."""
    cdef long long sa = (<long long *>a)[0]
    cdef long long sb = (<long long *>b)[0]
    if sa < sb:
        return -1
    if sa > sb:
        return 1
    cdef long long ea = (<long long *>a)[1]
    cdef long long eb = (<long long *>b)[1]
    if ea < eb:
        return -1
    if ea > eb:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(20000,))
def interval_overlap_count(int n):
    """Create n intervals, sort by start, count overlapping pairs via sweep."""
    # Allocate interleaved pairs array: pairs[i*2] = start, pairs[i*2+1] = end
    cdef long long *pairs = <long long *>malloc(n * 2 * sizeof(long long))
    if not pairs:
        raise MemoryError()

    # Allocate active_ends C array (worst case: all n ends are active)
    cdef long long *active = <long long *>malloc(n * sizeof(long long))
    if not active:
        free(pairs)
        raise MemoryError()

    cdef long long overlap_count = 0
    cdef long long start, length
    cdef int i, j, active_count, write_pos

    # Build intervals
    with nogil:
        for i in range(n):
            start = ((<long long>i * <long long>2654435761 + 17) ^ (<long long>i * <long long>1103515245)) % 10000
            length = ((<long long>i * <long long>1664525 + <long long>1013904223) ^ (<long long>i * <long long>214013)) % 500 + 1
            pairs[i * 2]     = start
            pairs[i * 2 + 1] = start + length

        # Sort by start (then end) using stdlib qsort
        qsort(pairs, n, 2 * sizeof(long long), _cmp_pairs)

        # Sweep line with in-place compaction of active_ends
        active_count = 0
        for i in range(n):
            start = pairs[i * 2]

            # Filter active_ends in-place: keep only those > current start
            write_pos = 0
            for j in range(active_count):
                if active[j] > start:
                    active[write_pos] = active[j]
                    write_pos += 1
            active_count = write_pos

            overlap_count += active_count
            active[active_count] = pairs[i * 2 + 1]
            active_count += 1

    free(pairs)
    free(active)
    return overlap_count
