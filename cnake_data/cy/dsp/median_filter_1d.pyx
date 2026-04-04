# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D median filter with configurable window size (Cython-optimized).

Keywords: dsp, median, filter, smoothing, signal processing, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def median_filter_1d(int n):
    """Apply 1D median filter with window size 7 to a deterministic signal."""
    cdef int i, j, a, b, count

    cdef int half_w = 3
    cdef double pi2 = 2.0 * M_PI
    cdef double checksum = 0.0, key
    cdef double *signal = <double *>malloc(n * sizeof(double))
    cdef double *output = <double *>malloc(n * sizeof(double))
    cdef double buf[7]
    if not signal or not output:
        if signal:
            free(signal)
        if output:
            free(output)
        raise MemoryError()

    # Generate signal
    for i in range(n):
        signal[i] = (sin(pi2 * i * 3.0 / n)
                      + 0.5 * cos(pi2 * i * 7.0 / n)
                      + (i * 17 % 53) / 53.0)

    # Apply median filter
    for i in range(n):
        count = 0
        for j in range(i - half_w, i + half_w + 1):
            if 0 <= j < n:
                buf[count] = signal[j]
                count += 1

        # Insertion sort
        for a in range(1, count):
            key = buf[a]
            b = a - 1
            while b >= 0 and buf[b] > key:
                buf[b + 1] = buf[b]
                b -= 1
            buf[b + 1] = key

        output[i] = buf[count // 2]

    # Compute checksum
    for i in range(n):
        checksum += output[i] * (i + 1)

    cdef double first = output[0]
    cdef double last = output[n - 1]
    free(signal)
    free(output)
    return (checksum, first, last)
