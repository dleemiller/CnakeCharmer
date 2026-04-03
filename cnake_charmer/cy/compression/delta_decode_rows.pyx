# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Delta decode a flattened 2D array row-by-row using prefix sums (Cython-optimized).

Keywords: compression, delta decoding, prefix sum, row-wise, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def delta_decode_rows(int n):
    """Delta decode h=n rows of width w=32 using malloc'd unsigned char array."""
    cdef int w = 32
    cdef int total = n * w
    cdef unsigned char *arr = <unsigned char *>malloc(total * sizeof(unsigned char))
    if not arr:
        raise MemoryError("Failed to allocate array")

    cdef int i, x, y, offset, pos
    cdef long long s
    cdef int mid_val, last_val

    # Initialize array
    for i in range(total):
        arr[i] = <unsigned char>((i * 7 + 13) % 256)

    # Delta decode each row
    with nogil:
        for y in range(n):
            offset = y * w
            for x in range(w - 1):
                pos = offset + x
                arr[pos + 1] = arr[pos + 1] + arr[pos]

    # Compute results
    s = 0
    for i in range(total):
        s += arr[i]
    mid_val = arr[total // 2]
    last_val = arr[total - 1]

    free(arr)

    return (int(s), int(mid_val), int(last_val))
