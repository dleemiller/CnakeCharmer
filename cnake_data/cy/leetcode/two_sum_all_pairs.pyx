# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find all index pairs that sum to a target value (Cython-optimized).

Keywords: leetcode, two sum, pairs, hash map, indices, cython, benchmark
"""

from libc.stdlib cimport malloc, free, calloc
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def two_sum_all_pairs(int n):
    """Find all pairs summing to target using typed variables and C frequency array."""
    cdef int target = n // 4
    cdef int half_n = n // 2
    if half_n == 0:
        half_n = 1

    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    # freq[v] = number of times value v has been seen so far
    # Values are in [0, half_n), so freq needs half_n entries
    cdef int *freq = <int *>calloc(half_n, sizeof(int))
    if not freq:
        free(arr)
        raise MemoryError()

    # first_seen[v] = earliest index at which value v appeared (-1 = unseen)
    cdef int *first_seen = <int *>malloc(half_n * sizeof(int))
    if not first_seen:
        free(arr)
        free(freq)
        raise MemoryError()

    cdef int i, j
    for i in range(half_n):
        first_seen[i] = -1

    for i in range(n):
        arr[i] = (i * 37 + 13) % half_n

    cdef long long num_pairs = 0
    cdef long long index_sum = 0
    cdef int first_pair_sum = -1
    cdef int complement, val, cnt

    for j in range(n):
        val = arr[j]
        complement = target - val
        # complement must be a valid index into freq
        if 0 <= complement < half_n:
            cnt = freq[complement]
            if cnt > 0:
                num_pairs += cnt
                index_sum += <long long>cnt * j
                if first_pair_sum == -1:
                    first_pair_sum = first_seen[complement] + j
        freq[val] += 1
        if first_seen[val] == -1:
            first_seen[val] = j

    free(arr)
    free(freq)
    free(first_seen)
    return (int(num_pairs), int(index_sum), first_pair_sum)
