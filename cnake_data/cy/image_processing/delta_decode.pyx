# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Delta decoding for image compression (zip-with-prediction) (Cython-optimized).

Keywords: image processing, delta decode, compression, prediction, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def delta_decode(int n):
    """Decode n×n delta-encoded image using C arrays and nogil."""
    cdef int x, y
    cdef unsigned char cumsum
    cdef long long total_sum
    cdef int size = n * n

    cdef unsigned char *encoded = <unsigned char *>malloc(size * sizeof(unsigned char))
    cdef unsigned char *decoded = <unsigned char *>malloc(size * sizeof(unsigned char))

    if not encoded or not decoded:
        if encoded: free(encoded)
        if decoded: free(decoded)
        raise MemoryError("Failed to allocate arrays")

    with nogil:
        # Generate encoded data
        for y in range(n):
            for x in range(n):
                encoded[y * n + x] = ((y * 1009 + x * 2003 + 42) * 17 + 137) & 0xFF

        # Delta decode each row
        for y in range(n):
            cumsum = 0
            for x in range(n):
                cumsum = (cumsum + encoded[y * n + x]) & 0xFF
                decoded[y * n + x] = cumsum

        # Compute total sum
        total_sum = 0
        for y in range(n):
            for x in range(n):
                total_sum += decoded[y * n + x]

    cdef long long ts = total_sum & 0xFFFFFFFF
    cdef int checksum = decoded[n // 2 * n + n // 3]

    free(encoded)
    free(decoded)
    return (ts, checksum)
