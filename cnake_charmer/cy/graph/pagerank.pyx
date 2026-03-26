# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""PageRank computation on a deterministic graph (Cython-optimized).

Keywords: graph, pagerank, ranking, iteration, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


cdef int _compare_doubles_desc(const void *a, const void *b) noexcept nogil:
    cdef double da = (<double *>a)[0]
    cdef double db = (<double *>b)[0]
    if da > db:
        return -1
    elif da < db:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(50000,))
def pagerank(int n):
    """Compute PageRank on n nodes for 20 iterations.

    Args:
        n: Number of nodes.

    Returns:
        Sum of top-10 PageRank values.
    """
    cdef double damping = 0.85
    cdef int iterations = 20
    cdef int out_degree = 2
    cdef int i, it, top_count
    cdef double contrib, base_rank, total

    cdef int *adj0 = <int *>malloc(n * sizeof(int))
    cdef int *adj1 = <int *>malloc(n * sizeof(int))
    cdef double *rank_arr = <double *>malloc(n * sizeof(double))
    cdef double *new_rank = <double *>malloc(n * sizeof(double))
    cdef double *tmp

    if not adj0 or not adj1 or not rank_arr or not new_rank:
        free(adj0)
        free(adj1)
        free(rank_arr)
        free(new_rank)
        raise MemoryError()

    for i in range(n):
        adj0[i] = (i * 3 + 1) % n
        adj1[i] = (i * 7 + 2) % n
        rank_arr[i] = 1.0 / n

    base_rank = (1.0 - damping) / n

    for it in range(iterations):
        for i in range(n):
            new_rank[i] = base_rank

        for i in range(n):
            contrib = damping * rank_arr[i] / out_degree
            new_rank[adj0[i]] += contrib
            new_rank[adj1[i]] += contrib

        tmp = rank_arr
        rank_arr = new_rank
        new_rank = tmp

    # Sort descending to find top 10
    qsort(rank_arr, n, sizeof(double), _compare_doubles_desc)

    top_count = 10
    if n < 10:
        top_count = n
    total = 0.0
    for i in range(top_count):
        total += rank_arr[i]

    free(adj0)
    free(adj1)
    free(rank_arr)
    free(new_rank)
    return total
