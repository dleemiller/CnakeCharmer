# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sum of Hamming distances between consecutive byte strings (Cython-optimized).

Keywords: string processing, hamming distance, byte comparison, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def hamming_distance_sum(int n):
    """Compute sum of Hamming distances using unsigned char* byte comparison."""
    cdef int i, j, k, dist
    cdef long long total = 0
    cdef int max_dist = 0
    cdef int str_len = 8
    cdef unsigned char *data = <unsigned char *>malloc(n * str_len * sizeof(unsigned char))
    cdef unsigned char *s1
    cdef unsigned char *s2

    if data == NULL:
        raise MemoryError("Failed to allocate array")

    # Generate all byte strings as flat array
    for i in range(n):
        for j in range(str_len):
            data[i * str_len + j] = (i * j + 3) % 256

    # Compare consecutive pairs
    for i in range(n - 1):
        s1 = &data[i * str_len]
        s2 = &data[(i + 1) * str_len]
        dist = 0
        for k in range(str_len):
            if s1[k] != s2[k]:
                dist += 1
        total += dist
        if dist > max_dist:
            max_dist = dist

    cdef long long result_total = total
    cdef int result_max = max_dist
    free(data)
    return (result_total, result_max)
