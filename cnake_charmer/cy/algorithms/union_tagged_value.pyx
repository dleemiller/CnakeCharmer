# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Tagged union pattern: process tagged int/double values and
accumulate sum (Cython-optimized).

Keywords: union, tagged, struct, algorithms, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cdef union ValueData:
    int i
    double d

cdef struct TaggedValue:
    int tag
    ValueData data


@cython_benchmark(syntax="cy", args=(100000,))
def union_tagged_value(int n):
    """Process n tagged values, sum them."""
    cdef double total = 0.0
    cdef int idx
    cdef unsigned long long h
    cdef TaggedValue tv

    for idx in range(n):
        h = (<unsigned long long>idx
             * <unsigned long long>2654435761)
        h = h & <unsigned long long>0xFFFFFFFF
        tv.tag = <int>(h % 2)
        if tv.tag == 0:
            tv.data.i = <int>(((h >> 8) & 0xFFFF) - 32768)
            total += tv.data.i
        else:
            tv.data.d = (
                ((h >> 8) & 0xFFFF)
                / 65535.0 * 200.0 - 100.0
            )
            total += tv.data.d
    return total
