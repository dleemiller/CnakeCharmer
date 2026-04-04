# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply a fixed 1D banded Toeplitz operator and summarize output energy (Cython)."""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef void _toeplitz_kernel(
    int* vals,
    int n,
    long long* total_out,
    long long* energy_out,
    int* mid_out,
) noexcept nogil:
    cdef int i, y
    cdef int a0 = 5
    cdef int a1 = -3
    cdef int a2 = 1
    cdef long long total = 0
    cdef long long energy = 0
    cdef int mid = 0

    for i in range(n):
        y = a0 * vals[i]
        if i > 0:
            y += a1 * vals[i - 1]
        if i + 1 < n:
            y += a1 * vals[i + 1]
        if i > 1:
            y += a2 * vals[i - 2]
        if i + 2 < n:
            y += a2 * vals[i + 2]
        total += y
        energy += y * y
        if i == n // 2:
            mid = y

    total_out[0] = total
    energy_out[0] = energy
    mid_out[0] = mid


@cython_benchmark(syntax="cy", args=(120000,))
def toeplitz_band_energy(int n):
    cdef int *vals
    cdef int i
    cdef long long total = 0
    cdef long long energy = 0
    cdef int mid = 0

    if n <= 0:
        return (0, 0, 0)

    vals = <int *>malloc(n * sizeof(int))
    if not vals:
        raise MemoryError()

    for i in range(n):
        vals[i] = ((i * 19 + 23) % 211) - 105

    with nogil:
        _toeplitz_kernel(vals, n, &total, &energy, &mid)

    free(vals)
    return (total, energy % 1000000007, mid)
