# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Weighted random-choice walk with deterministic LCG state (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 34eb7bf393034a3b474c6dffead875c6125c37fe
- filename: rng.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(32, 220000, 17))
def stack2_rng_weighted_walk(int bucket_count, int draw_count, int seed_offset):
    cdef unsigned int state = <unsigned int>((123456789 + seed_offset * 7919) & 0xFFFFFFFF)
    cdef int *weights = <int *>malloc(bucket_count * sizeof(int))
    cdef int *counts = <int *>malloc(bucket_count * sizeof(int))
    cdef int idx, step, val, top_idx = 0
    cdef unsigned int target
    cdef int total_weight = 0
    cdef int partial, pick
    cdef unsigned int checksum = 0

    if not weights or not counts:
        free(weights)
        free(counts)
        raise MemoryError()

    for idx in range(bucket_count):
        state = (1664525 * state + 1013904223)
        val = <int>((state >> 8) % 1000) + 1
        weights[idx] = val
        counts[idx] = 0
        total_weight += val

    for step in range(draw_count):
        state = (1664525 * state + 1013904223)
        target = state % <unsigned int>total_weight
        partial = 0
        pick = 0
        for idx in range(bucket_count):
            partial += weights[idx]
            if target < <unsigned int>partial:
                pick = idx
                break

        counts[pick] += 1
        checksum = (checksum + <unsigned int>((pick + 3) * (step + 11))) & 0xFFFFFFFF
        if counts[pick] > counts[top_idx]:
            top_idx = pick

    val = counts[top_idx]
    idx = counts[bucket_count - 1]
    free(weights)
    free(counts)
    return (top_idx, val, idx, checksum)
