# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based pseudo thread-local counters with weighted reduction (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef class ThreadLocal:
    cdef int *counters
    cdef int nslots

    def __cinit__(self, int nslots):
        cdef int i
        self.counters = <int *>malloc(nslots * sizeof(int))
        if not self.counters:
            raise MemoryError()
        self.nslots = nslots
        for i in range(nslots):
            self.counters[i] = 0

    def __dealloc__(self):
        if self.counters != NULL:
            free(self.counters)


cdef void _run_counters(ThreadLocal tl, int steps, int seed, int stride, long long *checksum_out, int *peak_out) noexcept nogil:
    cdef int i, slot, delta, v
    cdef long long checksum = 0
    cdef int peak = 0
    for i in range(steps):
        slot = (seed + i * stride) % tl.nslots
        delta = ((i * 13 + seed) % 7) - 3
        tl.counters[slot] += delta
        v = tl.counters[slot]
        checksum += v * ((slot & 3) + 1)
        if v > peak:
            peak = v
    checksum_out[0] = checksum
    peak_out[0] = peak


@cython_benchmark(syntax="cy", args=(64, 900000, 11, 5))
def threadlocal_counter_class(int nslots, int steps, int seed, int stride):
    cdef ThreadLocal tl = ThreadLocal(nslots)
    cdef long long checksum = 0
    cdef int peak = 0

    with nogil:
        _run_counters(tl, steps, seed, stride, &checksum, &peak)

    return (checksum, peak, tl.counters[nslots // 2])
