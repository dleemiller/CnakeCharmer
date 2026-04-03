# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""L1 ball projection via sort-and-threshold (Cython-optimized).

Keywords: optimization, projection, l1 norm, soft threshold, sorting, cython, benchmark
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


@cython_benchmark(syntax="cy", args=(80000,))
def l1_projection(int n):
    """Project n deterministic vectors of length 12 onto the L1 ball.

    Args:
        n: Number of vectors to process.

    Returns:
        Tuple of (result_sum, max_abs_after).
    """
    cdef int dim = 12
    cdef double radius = 3.0
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
    cdef double l1_norm, theta, cumsum, candidate, val, av, shrunk, av_after
    cdef int k, d, i

    for k in range(n):
        # Generate deterministic vector
        for d in range(dim):
            vec[d] = ((k * 17 + d * 31 + 5) % 2003) / 200.0 - 5.0

        # Compute L1 norm
        l1_norm = 0.0
        for d in range(dim):
            l1_norm += fabs(vec[d])

        # If already inside the L1 ball, no projection needed
        if l1_norm <= radius:
            for d in range(dim):
                result_sum += vec[d]
                av = fabs(vec[d])
                if av > max_abs_after:
                    max_abs_after = av
            continue

        # Sort absolute values descending
        for d in range(dim):
            abs_vals[d] = fabs(vec[d])
        qsort(<void *>abs_vals, <size_t>dim, sizeof(double), _compare_desc)

        # Find theta
        cumsum = 0.0
        theta = 0.0
        for i in range(dim):
            cumsum += abs_vals[i]
            candidate = (cumsum - radius) / (i + 1)
            if i < dim - 1 and candidate >= abs_vals[i + 1]:
                theta = candidate
                break
            if i == dim - 1:
                theta = candidate

        # Apply soft-thresholding
        for d in range(dim):
            val = vec[d]
            av = fabs(val)
            if av <= theta:
                shrunk = 0.0
            elif val > 0.0:
                shrunk = val - theta
            else:
                shrunk = val + theta

            result_sum += shrunk
            av_after = fabs(shrunk)
            if av_after > max_abs_after:
                max_abs_after = av_after

    free(vec)
    free(abs_vals)
    return (result_sum, max_abs_after)
