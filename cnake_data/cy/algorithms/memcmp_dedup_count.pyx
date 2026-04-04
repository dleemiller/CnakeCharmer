# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort 8-byte records and count unique via memcmp.

Keywords: algorithms, memcmp, dedup, sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcmp
from cnake_data.benchmarks import cython_benchmark


cdef int cmp_records(
    const void *a, const void *b
) noexcept nogil:
    """Compare two 8-byte records."""
    return memcmp(a, b, 8)


@cython_benchmark(syntax="cy", args=(50000,))
def memcmp_dedup_count(int n):
    """Sort n 8-byte records, count unique via memcmp."""
    cdef int rec_len = 8
    cdef unsigned char *data = <unsigned char *>malloc(
        n * rec_len
    )
    if not data:
        raise MemoryError()

    cdef int i, j, unique
    cdef long long seed

    for i in range(n):
        seed = (
            <long long>i * <long long>2654435761 + 17
        )
        for j in range(rec_len):
            seed = (
                seed * 1103515245 + 12345
            ) & 0x7FFFFFFF
            data[i * rec_len + j] = seed % 256

    qsort(data, n, rec_len, cmp_records)

    unique = 1
    for i in range(1, n):
        if memcmp(
            &data[i * rec_len],
            &data[(i - 1) * rec_len],
            rec_len,
        ) != 0:
            unique += 1

    free(data)
    return unique
