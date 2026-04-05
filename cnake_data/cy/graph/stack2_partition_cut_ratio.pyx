# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute partition cut and ratio metrics on deterministic weighted graphs (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 7035f318b0301a1b1758627b664e7ddba244321f
- filename: tools.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(620, 9, 29))
def stack2_partition_cut_ratio(int node_count, int group_mod, int seed_tag):
    cdef int *degrees
    cdef int left, right, weight, left_group, right_group
    cdef int cross_sum = 0
    cdef int internal_sum = 0
    cdef int max_degree = 0
    cdef int checksum
    cdef int ratio_scaled

    if group_mod <= 0:
        return (0, 0, 0, 0)

    degrees = <int *>malloc(node_count * sizeof(int))
    if not degrees:
        raise MemoryError()

    for left in range(node_count):
        degrees[left] = 0

    for left in range(node_count):
        left_group = left % group_mod
        for right in range(left + 1, node_count):
            weight = (left * 131 + right * 17 + seed_tag * 19) % 31
            if weight < 6:
                continue

            right_group = right % group_mod
            degrees[left] += weight
            degrees[right] += weight
            if left_group == right_group:
                internal_sum += weight
            else:
                cross_sum += weight

    for left in range(node_count):
        if degrees[left] > max_degree:
            max_degree = degrees[left]

    ratio_scaled = (<long long>cross_sum * 1000000) // (internal_sum + 1)
    checksum = (cross_sum * 97 + internal_sum * 53 + max_degree * 11) & 0xFFFFFFFF
    free(degrees)
    return (cross_sum, internal_sum, ratio_scaled, checksum)
