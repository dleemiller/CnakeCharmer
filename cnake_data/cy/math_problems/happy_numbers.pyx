# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find all happy numbers up to n (Cython-optimized).

Keywords: happy, numbers, digit, squares, math, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(100000,))
def happy_numbers(int n):
    """Count happy numbers from 1 to n and find the largest one.

    Uses a cache array to memoize happy/unhappy status for previously seen numbers.
    """
    cdef int num, val, s, d, tmp
    cdef int count = 0
    cdef int last_happy = 0
    cdef long long checksum = 0

    # Cache: 0 = unknown, 1 = happy, 2 = unhappy
    cdef int cache_size = n + 1 if n >= 1000 else 1001
    cdef char *cache = <char *>malloc(cache_size * sizeof(char))
    if cache == NULL:
        raise MemoryError()

    cdef int i
    for i in range(cache_size):
        cache[i] = 0

    # Visited buffer for cycle detection (max digits for int is ~10, sum of squares
    # of digits of a 6-digit number is at most 9^2*6 = 486, so cycle values < 1000)
    cdef int *visited = <int *>malloc(1000 * sizeof(int))
    if visited == NULL:
        free(cache)
        raise MemoryError()

    cdef int visit_count, j
    cdef int is_happy

    for num in range(1, n + 1):
        val = num
        visit_count = 0

        # Walk the chain until we hit 1, a cached result, or detect a cycle
        while val != 1:
            if val < cache_size and cache[val] != 0:
                if cache[val] == 1:
                    val = 1
                break
            # Check if we've seen this value in current chain
            is_happy = 0
            for j in range(visit_count):
                if visited[j] == val:
                    is_happy = -1
                    break
            if is_happy == -1:
                break

            if visit_count < 1000:
                visited[visit_count] = val
                visit_count += 1

            s = 0
            tmp = val
            while tmp > 0:
                d = tmp % 10
                s += d * d
                tmp = tmp // 10
            val = s

        if val == 1:
            count += 1
            last_happy = num
            checksum = (checksum + num) % MOD
            # Mark all visited as happy
            for j in range(visit_count):
                if visited[j] < cache_size:
                    cache[visited[j]] = 1
        else:
            # Mark all visited as unhappy
            for j in range(visit_count):
                if visited[j] < cache_size:
                    cache[visited[j]] = 2

    free(cache)
    free(visited)
    return (count, last_happy, int(checksum))
