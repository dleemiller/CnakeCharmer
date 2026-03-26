# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gale-Shapley stable marriage algorithm.

Keywords: algorithms, stable marriage, gale shapley, matching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def stable_marriage(int n):
    """Run Gale-Shapley stable matching for n men and n women."""
    cdef int *man_pref = <int *>malloc(n * n * sizeof(int))
    cdef int *woman_rank = <int *>malloc(n * n * sizeof(int))
    cdef int *man_next = <int *>malloc(n * sizeof(int))
    cdef int *woman_partner = <int *>malloc(n * sizeof(int))
    cdef int *man_partner = <int *>malloc(n * sizeof(int))
    cdef int *free_stack = <int *>malloc(n * sizeof(int))

    if not man_pref or not woman_rank or not man_next or not woman_partner or not man_partner or not free_stack:
        if man_pref: free(man_pref)
        if woman_rank: free(woman_rank)
        if man_next: free(man_next)
        if woman_partner: free(woman_partner)
        if man_partner: free(man_partner)
        if free_stack: free(free_stack)
        raise MemoryError()

    cdef int i, j, m, w, old_m
    cdef int free_top = n
    cdef long long total = 0

    # Build man preferences
    for i in range(n):
        for j in range(n):
            man_pref[i * n + j] = (i + j * 7) % n

    # Build woman rankings
    for w in range(n):
        for j in range(n):
            m = (w + j * 13) % n
            woman_rank[w * n + m] = j

    # Initialize
    for i in range(n):
        man_next[i] = 0
        woman_partner[i] = -1
        man_partner[i] = -1
        free_stack[i] = i

    # Gale-Shapley
    while free_top > 0:
        free_top -= 1
        m = free_stack[free_top]
        w = man_pref[m * n + man_next[m]]
        man_next[m] += 1

        if woman_partner[w] == -1:
            woman_partner[w] = m
            man_partner[m] = w
        elif woman_rank[w * n + m] < woman_rank[w * n + woman_partner[w]]:
            old_m = woman_partner[w]
            woman_partner[w] = m
            man_partner[m] = w
            man_partner[old_m] = -1
            free_stack[free_top] = old_m
            free_top += 1
        else:
            free_stack[free_top] = m
            free_top += 1

    # Sum partner indices
    for i in range(n):
        total += man_partner[i]

    free(man_pref)
    free(woman_rank)
    free(man_next)
    free(woman_partner)
    free(man_partner)
    free(free_stack)

    return int(total)
