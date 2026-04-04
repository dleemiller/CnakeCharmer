# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort fixed-size blocks with insertion sort on stack C array.

Keywords: sorting, insertion sort, stack array, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def stack_array_sort(int n):
    """Sort 1024-element blocks, return checksum."""
    cdef int arr[1024]
    cdef int iterations = n // 1024
    cdef long long checksum = 0
    cdef long long seed
    cdef int i, j, key, it

    if iterations < 1:
        iterations = 1

    for it in range(iterations):
        seed = (
            <long long>it * <long long>2654435761 + 17
        )
        for i in range(1024):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            arr[i] = seed % 100000

        # Insertion sort
        for i in range(1, 1024):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        for i in range(0, 1024, 64):
            checksum += arr[i]

    return checksum
