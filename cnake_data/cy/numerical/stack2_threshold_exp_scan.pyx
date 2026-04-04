# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Thresholded exponential scan across generated scalar series (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: c5a83df9c1b6b624fa118d7e7eb14b6acd9ea353
- filename: dot_product.pyx
"""

from libc.math cimport exp
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2200000, 620, 13))
def stack2_threshold_exp_scan(int vector_size, int threshold_milli, int seed_tag):
    cdef unsigned int state = <unsigned int>((987123 + seed_tag * 4129) & 0x7FFFFFFF)
    cdef double threshold = threshold_milli / 1000.0
    cdef int idx
    cdef double val
    cdef int scaled
    cdef int active = 0
    cdef unsigned int total_scaled = 0
    cdef unsigned int checksum = 0
    cdef int last_scaled = 0

    for idx in range(vector_size):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        val = (state & 0xFFFF) / 65535.0
        if val > threshold:
            scaled = <int>(exp(val) * 1000.0)
            active += 1
            total_scaled = (total_scaled + <unsigned int>scaled) & 0xFFFFFFFF
            last_scaled = scaled
            checksum = (checksum + <unsigned int>(scaled * (idx + 1))) & 0xFFFFFFFF

    return (active, total_scaled, checksum, last_scaled)
