# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count bounded integer compositions and aggregate occupancy probabilities (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef double _count_ways(int cells, int mines, int cap) noexcept:
    cdef double *dp = <double *>malloc((mines + 1) * sizeof(double))
    cdef double *nxt = <double *>malloc((mines + 1) * sizeof(double))
    cdef int i, used, add, max_add
    cdef double result = 0.0

    if not dp or not nxt:
        free(dp)
        free(nxt)
        return 0.0

    for i in range(mines + 1):
        dp[i] = 0.0
    dp[0] = 1.0

    for i in range(cells):
        for used in range(mines + 1):
            nxt[used] = 0.0
        for used in range(mines + 1):
            if dp[used] == 0.0:
                continue
            max_add = cap
            if used + max_add > mines:
                max_add = mines - used
            for add in range(max_add + 1):
                nxt[used + add] += dp[used]
        for used in range(mines + 1):
            dp[used] = nxt[used]

    result = dp[mines]
    free(dp)
    free(nxt)
    return result


@cython_benchmark(syntax="cy", args=(36, 54, 3, 24))
def bounded_mine_prob(int num_cells, int total_mines, int max_per_cell, int repeats):
    cdef int r, mines
    cdef double total, without_first, prob
    cdef double ways_sum = 0.0
    cdef double prob_sum = 0.0
    cdef double last_prob = 0.0

    for r in range(repeats):
        mines = total_mines - (r % (max_per_cell + 1))
        if mines < 0:
            mines = 0
        total = _count_ways(num_cells, mines, max_per_cell)
        without_first = _count_ways(num_cells - 1, mines, max_per_cell)
        if total == 0.0:
            prob = 0.0
        else:
            prob = 1.0 - without_first / total
        ways_sum += total
        prob_sum += prob
        last_prob = prob

    return (ways_sum, prob_sum, last_prob)
