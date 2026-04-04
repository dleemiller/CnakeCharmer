# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute group statistics using struct-dict roundtrip.

Demonstrates struct auto-conversion to dict: Stats struct
is returned from cdef helper and auto-converts at boundary.

Keywords: algorithms, struct, dict, roundtrip, statistics, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


cdef struct Stats:
    double mean
    double std
    int count


cdef Stats _compute_stats(
    int group_id, int group_size,
) noexcept:
    """Compute stats for a group of hash-derived values."""
    cdef Stats st
    cdef double s = 0.0
    cdef double s2 = 0.0
    cdef double val
    cdef int k, idx
    cdef unsigned int h

    for k in range(group_size):
        idx = group_id * group_size + k
        h = (
            (<unsigned int>idx
             * <unsigned int>2654435761)
            ^ (<unsigned int>idx
               * <unsigned int>2246822519)
        )
        val = <double>(h & 0xFFFF) / 65535.0
        s += val
        s2 += val * val

    st.mean = s / <double>group_size
    st.std = sqrt(
        s2 / <double>group_size - st.mean * st.mean
    )
    st.count = group_size
    return st


@cython_benchmark(syntax="cy", args=(10000,))
def struct_dict_roundtrip(int n):
    """Compute stats for n groups, sum of means."""
    cdef int g
    cdef int group_size = 10
    cdef double total_mean = 0.0
    cdef Stats st

    for g in range(n):
        st = _compute_stats(g, group_size)
        total_mean += st.mean

    return total_mean
