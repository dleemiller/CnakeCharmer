# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""L-infinity proximal operator applied to deterministic vectors (Cython-optimized).

Keywords: optimization, proximal, l-infinity, projection, sorting, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef int _compare_desc(const void *a, const void *b) noexcept nogil:
    """Compare doubles in descending order for qsort."""
    cdef double va = (<double *>a)[0]
    cdef double vb = (<double *>b)[0]
    if va > vb:
        return -1
    elif va < vb:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(50000,))
def linf_proximal(int n):
    """Apply L-inf proximal operator to n deterministic vectors of length 8.

    Args:
        n: Number of vectors to process.

    Returns:
        Tuple of (result_sum, max_abs_after).
    """
    cdef int dim = 8
    cdef double tau = 1.5
    cdef double *vec = <double *>malloc(dim * sizeof(double))
    cdef double *abs_vals = <double *>malloc(dim * sizeof(double))
    if not vec or not abs_vals:
        if vec:
            free(vec)
        if abs_vals:
            free(abs_vals)
        raise MemoryError()

    cdef double result_sum = 0.0
    cdef double max_abs_after = 0.0
    cdef double lam, cumsum, candidate, val, av, shrunk, av_after
    cdef int k, d, i

    for k in range(n):
        # Generate deterministic vector
        for d in range(dim):
            vec[d] = ((k * 7 + d * 13 + 3) % 1000) / 100.0 - 5.0

        # Sort absolute values descending
        for d in range(dim):
            abs_vals[d] = fabs(vec[d])
        qsort(<void *>abs_vals, <size_t>dim, sizeof(double), _compare_desc)

        # Find threshold
        lam = 0.0
        cumsum = 0.0
        for i in range(dim):
            cumsum += abs_vals[i]
            candidate = (cumsum - tau) / (i + 1)
            if candidate > abs_vals[i]:
                break
            if candidate > lam:
                lam = candidate

        # Apply soft-thresholding
        for d in range(dim):
            val = vec[d]
            av = fabs(val)
            if av <= lam:
                shrunk = 0.0
            elif val > 0:
                shrunk = val - lam
            else:
                shrunk = val + lam

            result_sum += shrunk
            av_after = fabs(shrunk)
            if av_after > max_abs_after:
                max_abs_after = av_after

    free(vec)
    free(abs_vals)
    return (result_sum, max_abs_after)
