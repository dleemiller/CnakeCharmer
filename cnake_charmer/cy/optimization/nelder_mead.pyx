# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""N-dimensional Nelder-Mead optimizer (Cython-optimized).

Keywords: nelder-mead, simplex, optimization, n-dimensional, rosenbrock, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef double _rosenbrock(double *x, int dim) noexcept nogil:
    """Rosenbrock function in N dimensions."""
    cdef double val = 0.0
    cdef double t1, t2
    cdef int i
    for i in range(dim - 1):
        t1 = x[i + 1] - x[i] * x[i]
        t2 = 1.0 - x[i]
        val += 100.0 * t1 * t1 + t2 * t2
    return val


@cython_benchmark(syntax="cy", args=(4, 12))
def nelder_mead(int dim, int n_starts):
    """N-dimensional Nelder-Mead minimization of the Rosenbrock function."""
    cdef double alpha = 1.0
    cdef double gamma = 2.0
    cdef double rho = 0.5
    cdef double sig = 0.5
    cdef int max_iter = 200 * dim
    cdef double step = 0.5
    cdef double global_best = 1e30
    cdef double checksum = 0.0
    cdef int n_verts = dim + 1
    cdef int s, d, i, j, _it
    cdef double inv_n = 1.0 / dim
    cdef double key_f, f_r, f_e, f_c, f_worst, best_f
    cdef double *key_v

    # Allocate simplex: n_verts rows x dim columns (flat)
    cdef double *simplex = <double *>malloc(n_verts * dim * sizeof(double))
    cdef double *f_vals = <double *>malloc(n_verts * sizeof(double))
    cdef double *centroid = <double *>malloc(dim * sizeof(double))
    cdef double *xr = <double *>malloc(dim * sizeof(double))
    cdef double *xe = <double *>malloc(dim * sizeof(double))
    cdef double *xc = <double *>malloc(dim * sizeof(double))
    # Row pointers for simplex (to allow swapping)
    cdef double **rows = <double **>malloc(n_verts * sizeof(double *))

    if not simplex or not f_vals or not centroid or not xr or not xe or not xc or not rows:
        free(simplex); free(f_vals); free(centroid)
        free(xr); free(xe); free(xc); free(rows)
        return (0.0, 0.0)

    for i in range(n_verts):
        rows[i] = simplex + i * dim

    for s in range(n_starts):
        # Build initial simplex
        for d in range(dim):
            rows[0][d] = ((s * 137 + d * 43) % 1000) / 250.0 - 2.0
        f_vals[0] = _rosenbrock(rows[0], dim)

        for i in range(1, n_verts):
            for d in range(dim):
                rows[i][d] = rows[0][d]
            rows[i][i - 1] = rows[0][i - 1] + step
            f_vals[i] = _rosenbrock(rows[i], dim)

        for _it in range(max_iter):
            # Insertion sort by f_vals (swap row pointers)
            for i in range(1, n_verts):
                key_f = f_vals[i]
                key_v = rows[i]
                j = i - 1
                while j >= 0 and f_vals[j] > key_f:
                    f_vals[j + 1] = f_vals[j]
                    rows[j + 1] = rows[j]
                    j -= 1
                f_vals[j + 1] = key_f
                rows[j + 1] = key_v

            # Centroid of all but worst
            for d in range(dim):
                centroid[d] = 0.0
            for i in range(n_verts - 1):
                for d in range(dim):
                    centroid[d] += rows[i][d]
            for d in range(dim):
                centroid[d] *= inv_n

            f_worst = f_vals[dim]

            # Reflection
            for d in range(dim):
                xr[d] = centroid[d] + alpha * (centroid[d] - rows[dim][d])
            f_r = _rosenbrock(xr, dim)

            if f_vals[0] <= f_r and f_r < f_vals[dim - 1]:
                for d in range(dim):
                    rows[dim][d] = xr[d]
                f_vals[dim] = f_r
                continue

            # Expansion
            if f_r < f_vals[0]:
                for d in range(dim):
                    xe[d] = centroid[d] + gamma * (xr[d] - centroid[d])
                f_e = _rosenbrock(xe, dim)
                if f_e < f_r:
                    for d in range(dim):
                        rows[dim][d] = xe[d]
                    f_vals[dim] = f_e
                else:
                    for d in range(dim):
                        rows[dim][d] = xr[d]
                    f_vals[dim] = f_r
                continue

            # Contraction
            if f_r < f_worst:
                # Outside contraction
                for d in range(dim):
                    xc[d] = centroid[d] + rho * (xr[d] - centroid[d])
                f_c = _rosenbrock(xc, dim)
                if f_c <= f_r:
                    for d in range(dim):
                        rows[dim][d] = xc[d]
                    f_vals[dim] = f_c
                    continue
            else:
                # Inside contraction
                for d in range(dim):
                    xc[d] = centroid[d] + rho * (rows[dim][d] - centroid[d])
                f_c = _rosenbrock(xc, dim)
                if f_c < f_worst:
                    for d in range(dim):
                        rows[dim][d] = xc[d]
                    f_vals[dim] = f_c
                    continue

            # Shrink
            for i in range(1, n_verts):
                for d in range(dim):
                    rows[i][d] = rows[0][d] + sig * (rows[i][d] - rows[0][d])
                f_vals[i] = _rosenbrock(rows[i], dim)

        best_f = f_vals[0]
        checksum += best_f * (s + 1)
        if best_f < global_best:
            global_best = best_f

    free(simplex)
    free(f_vals)
    free(centroid)
    free(xr)
    free(xe)
    free(xc)
    free(rows)

    return (global_best, checksum)
