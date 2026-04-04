# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Binary search with except -1 error return spec (Cython-optimized).

Keywords: algorithms, binary search, error handling, except spec, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef int _binary_search(int *arr, int n, int target) except -1:
    """Binary search returning index or -1 if not found.

    Uses except -1 error return specification.
    """
    cdef int lo = 0
    cdef int hi = n - 1
    cdef int mid
    while lo <= hi:
        mid = (lo + hi) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # Not found - but we can't return -1 as that would trigger exception check.
    # Use a sentinel approach: return -(n+1) to indicate not found.
    return -(n + 1)


@cython_benchmark(syntax="cy", args=(100000,))
def except_value_search(int n):
    """Perform n binary searches using cdef with except -1 spec."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        arr[i] = i * 3

    cdef int found_count = 0
    cdef unsigned long long hash_val
    cdef int target, idx

    for i in range(n):
        hash_val = <unsigned long long>i * <unsigned long long>2654435761
        hash_val = hash_val & <unsigned long long>0xFFFFFFFF
        target = <int>(hash_val % <unsigned long long>(n * 4))
        idx = _binary_search(arr, n, target)
        if idx >= 0:
            found_count += 1

    free(arr)
    return found_count
