# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Flip an image and compute checksum (Cython with typed memoryviews).

Keywords: image, flip, typed memoryview, transform, checksum, image processing, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def image_flip_checksum(int n):
    """Flip image using typed memoryviews and compute weighted checksum."""
    cdef int nn = n * n
    cdef unsigned char *img_ptr = <unsigned char *>malloc(nn * sizeof(unsigned char))
    cdef unsigned char *flip_ptr = <unsigned char *>malloc(nn * sizeof(unsigned char))
    cdef unsigned char *res_ptr = <unsigned char *>malloc(nn * sizeof(unsigned char))
    if not img_ptr or not flip_ptr or not res_ptr:
        if img_ptr: free(img_ptr)
        if flip_ptr: free(flip_ptr)
        if res_ptr: free(res_ptr)
        raise MemoryError()

    # Create typed memoryviews from raw pointers
    cdef unsigned char[:] img = <unsigned char[:nn]>img_ptr
    cdef unsigned char[:] flipped = <unsigned char[:nn]>flip_ptr
    cdef unsigned char[:] result = <unsigned char[:nn]>res_ptr

    cdef int i, j
    cdef unsigned long long h
    cdef long long checksum = 0

    # Generate image
    for i in range(n):
        for j in range(n):
            h = ((<unsigned long long>i * 2654435761 + <unsigned long long>j * 1103515245 + 7) >> 4) & 0xFF
            img[i * n + j] = <unsigned char>h

    # Horizontal flip using memoryview access
    for i in range(n):
        for j in range(n):
            flipped[i * n + j] = img[i * n + (n - 1 - j)]

    # Vertical flip using memoryview access
    for i in range(n):
        for j in range(n):
            result[i * n + j] = flipped[(n - 1 - i) * n + j]

    # Weighted checksum
    for i in range(nn):
        checksum += <long long>result[i] * ((i % 256) + 1)
        checksum = checksum & 0x7FFFFFFF

    free(img_ptr)
    free(flip_ptr)
    free(res_ptr)
    return checksum
