# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""String ops using cdef extern from for strlen and strcmp.

Keywords: string processing, extern, strlen, strcmp, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "string.h":
    size_t strlen(const char *s) nogil
    int strcmp(const char *a, const char *b) nogil


@cython_benchmark(syntax="cy", args=(50000,))
def extern_string_ops(int n):
    """Generate n strings, total length + equal pairs."""
    cdef int max_len = 9  # 8 chars + null
    cdef char *buf = <char *>malloc(
        n * max_len
    )
    if not buf:
        raise MemoryError()

    cdef int i, j, slen
    cdef long long seed
    cdef long long length_sum = 0
    cdef int equal_count = 0
    cdef char *p

    for i in range(n):
        seed = (
            <long long>i * <long long>2654435761 + 17
        )
        slen = (seed & 0x7FFFFFFF) % 8 + 1
        p = &buf[i * max_len]
        for j in range(slen):
            seed = (
                seed * 1103515245 + 12345
            ) & 0x7FFFFFFF
            p[j] = 65 + seed % 26
        p[slen] = 0  # null terminate

    for i in range(n):
        length_sum += strlen(&buf[i * max_len])

    for i in range(1, n):
        if strcmp(
            &buf[i * max_len],
            &buf[(i - 1) * max_len],
        ) == 0:
            equal_count += 1

    free(buf)
    return length_sum + equal_count
