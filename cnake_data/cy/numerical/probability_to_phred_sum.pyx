# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Convert probabilities to Phred scores and accumulate (Cython)."""

from libc.math cimport log10

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(48271, 200000, 1e-6))
def probability_to_phred_sum(int seed, int samples, double floor):
    return _probability_to_phred_sum_impl(seed, samples, floor)


cdef double _probability_to_phred_sum_impl(int seed, int samples, double floor) noexcept:
    cdef int i
    cdef int state = seed & 0x7FFFFFFF
    cdef double p
    cdef double total = 0.0
    for i in range(samples):
        state = <int>((<long long>state * 48271) % 2147483647)
        p = floor + (1.0 - floor) * (state / 2147483647.0)
        total += -10.0 * log10(p)
    return total
