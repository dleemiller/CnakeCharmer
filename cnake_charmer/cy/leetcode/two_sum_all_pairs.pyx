# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find all index pairs that sum to a target value (Cython-optimized).

Keywords: leetcode, two sum, pairs, hash map, indices, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def two_sum_all_pairs(int n):
    """Find all pairs summing to target using typed variables and hash map."""
    cdef int target = n // 4
    cdef int half_n = n // 2
    if half_n == 0:
        half_n = 1

    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j
    for i in range(n):
        arr[i] = (i * 37 + 13) % half_n

    # Use Python dict for hash map (Cython still benefits from typed loop vars)
    cdef long long num_pairs = 0
    cdef long long index_sum = 0
    cdef int first_pair_sum = -1
    cdef int complement, count

    index_map = {}

    for j in range(n):
        complement = target - arr[j]
        if complement in index_map:
            count = index_map[complement]
            num_pairs += count
            index_sum += <long long>count * j
        if arr[j] in index_map:
            index_map[arr[j]] = index_map[arr[j]] + 1
        else:
            index_map[arr[j]] = 1

    # Find first pair
    seen = {}
    for j in range(n):
        complement = target - arr[j]
        if complement in seen:
            first_pair_sum = seen[complement] + j
            break
        if arr[j] not in seen:
            seen[arr[j]] = j

    free(arr)
    return (int(num_pairs), int(index_sum), first_pair_sum)
