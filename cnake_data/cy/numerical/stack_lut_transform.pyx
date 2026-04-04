# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply lookup table transform using stack-allocated C array.

Keywords: numerical, lookup table, sin, stack array, cython, benchmark
"""

from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def stack_lut_transform(int n):
    """Build sin LUT on stack and apply to n values."""
    cdef double lut[256]
    cdef int i
    cdef long long idx
    cdef double total = 0.0

    for i in range(256):
        lut[i] = sin(i * 0.0245436926)

    for i in range(n):
        idx = (
            (<long long>i * <long long>2654435761) >> 4
        ) & 255
        total += lut[idx]

    return total
