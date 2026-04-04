# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Validate arrays using except * error return spec.

Uses `cdef void validate(...) except *:` which means Cython
always checks for Python exception after calling the function.

Keywords: algorithms, validation, error handling, except star, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef void _validate(int *arr, int n, int threshold) except *:
    """Validate array: no negatives, sum < threshold.

    Raises ValueError on invalid data. The except * spec
    tells Cython to always check for Python exceptions.
    """
    cdef int i
    cdef int total = 0
    for i in range(n):
        if arr[i] < 0:
            raise ValueError("negative element")
        total += arr[i]
    if total > threshold:
        raise ValueError("sum exceeds threshold")


@cython_benchmark(syntax="cy", args=(50000,))
def except_star_validate(int n):
    """Validate n arrays, count valid ones."""
    cdef int i, k, idx
    cdef int arr_size = 8
    cdef int threshold = 3000
    cdef int valid_count = 0
    cdef unsigned int h

    cdef int *arr = <int *>malloc(
        arr_size * sizeof(int)
    )
    if not arr:
        raise MemoryError()

    for i in range(n):
        for k in range(arr_size):
            idx = i * arr_size + k
            h = (
                (<unsigned int>idx
                 * <unsigned int>2654435761)
                ^ (<unsigned int>idx
                   * <unsigned int>2246822519)
            )
            arr[k] = <int>(h & 0xFFF) - 500

        try:
            _validate(arr, arr_size, threshold)
            valid_count += 1
        except ValueError:
            pass

    free(arr)
    return valid_count
