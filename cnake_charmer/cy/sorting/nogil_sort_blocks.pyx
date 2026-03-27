# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort blocks of elements with GIL release, returning checksum.

Keywords: sorting, insertion sort, nogil, blocks, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef void _insertion_sort(
    int *arr, int n
) noexcept nogil:
    """Sort n elements in-place using insertion sort."""
    cdef int i, j, key
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@cython_benchmark(syntax="cy", args=(10000,))
def nogil_sort_blocks(int n):
    """Sort n blocks of 256 elements, return checksum."""
    cdef int block_size = 256
    cdef int *block = <int *>malloc(
        block_size * sizeof(int)
    )
    if not block:
        raise MemoryError()

    cdef unsigned long long checksum = 0
    cdef int b, i
    cdef unsigned int v

    for b in range(n):
        for i in range(block_size):
            v = <unsigned int>(
                (<unsigned int>b * 7919u
                + <unsigned int>i * 104729u + 31u)
                & 0x7FFFFFFFu
            )
            v = <unsigned int>(
                ((v ^ (v >> 16)) * 73244475u)
                & 0x7FFFFFFFu
            )
            block[i] = <int>v

        with nogil:
            _insertion_sort(block, block_size)

        checksum += <unsigned long long>(
            <long long>block[0]
            + <long long>block[block_size // 2]
            + <long long>block[block_size - 1]
        )
        checksum = checksum & 0x7FFFFFFFFFFFFFFF

    free(block)
    return checksum
