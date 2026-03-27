# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Levenshtein edit distance (Cython-optimized).

Keywords: string processing, edit distance, levenshtein, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.stdlib cimport malloc, free


@cython_benchmark(syntax="cy", args=(500,))
def edit_distance(int n):
    """Compute Levenshtein edit distance using flat C arrays."""
    cdef str s1 = "ab" * n
    cdef str s2 = "ba" * n
    cdef int len1 = len(s1)
    cdef int len2 = len(s2)
    cdef int i, j, replace_cost, insert_cost, delete_cost
    cdef int *prev = <int *>malloc((len2 + 1) * sizeof(int))
    cdef int *curr = <int *>malloc((len2 + 1) * sizeof(int))
    cdef int *tmp

    if prev == NULL or curr == NULL:
        if prev != NULL:
            free(prev)
        if curr != NULL:
            free(curr)
        raise MemoryError("Failed to allocate DP arrays")

    for j in range(len2 + 1):
        prev[j] = j

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                delete_cost = prev[j]
                insert_cost = curr[j - 1]
                replace_cost = prev[j - 1]
                curr[j] = 1 + min(delete_cost, insert_cost, replace_cost)
        tmp = prev
        prev = curr
        curr = tmp

    cdef int result = prev[len2]
    cdef int result_mid = prev[len2 // 2]
    free(prev)
    free(curr)
    return (result, result_mid)
