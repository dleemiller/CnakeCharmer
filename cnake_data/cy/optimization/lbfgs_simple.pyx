# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simplified L-BFGS optimization (Cython-optimized).

Keywords: lbfgs, quasi-newton, optimization, minimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def lbfgs_simple(int n):
    """L-BFGS minimization of sum_i (x_i - i)^2 + 0.1*sum(x_i * x_{i+1})."""
    cdef int m = 5  # history size
    cdef int iteration, i, j, hist_len
    cdef double step, gamma, sy_val, yy_val, dot_val, beta_val, sy_dot
    cdef double obj, diff
    cdef double *tmp_s
    cdef double *tmp_y

    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *g = <double *>malloc(n * sizeof(double))
    cdef double *q = <double *>malloc(n * sizeof(double))
    cdef double *x_old = <double *>malloc(n * sizeof(double))
    cdef double *g_old = <double *>malloc(n * sizeof(double))
    cdef double *g_new = <double *>malloc(n * sizeof(double))
    # History: m entries, each of size n
    cdef double **s_hist = <double **>malloc(m * sizeof(double *))
    cdef double **y_hist = <double **>malloc(m * sizeof(double *))
    cdef double *rho_hist = <double *>malloc(m * sizeof(double))
    cdef double *alpha_list = <double *>malloc(m * sizeof(double))

    if not x or not g or not q or not x_old or not g_old or not g_new or not s_hist or not y_hist or not rho_hist or not alpha_list:
        raise MemoryError()

    for j in range(m):
        s_hist[j] = <double *>malloc(n * sizeof(double))
        y_hist[j] = <double *>malloc(n * sizeof(double))
        if not s_hist[j] or not y_hist[j]:
            raise MemoryError()

    # Initialize
    for i in range(n):
        x[i] = 0.0

    hist_len = 0

    for iteration in range(100):
        # Compute gradient
        for i in range(n):
            g[i] = 2.0 * (x[i] - i)
        for i in range(n - 1):
            g[i] += 0.1 * x[i + 1]
            g[i + 1] += 0.1 * x[i]

        # Copy q = g
        for i in range(n):
            q[i] = g[i]

        # L-BFGS two-loop: backward
        for j in range(hist_len - 1, -1, -1):
            dot_val = 0.0
            for i in range(n):
                dot_val += s_hist[j][i] * q[i]
            alpha_list[j] = rho_hist[j] * dot_val
            for i in range(n):
                q[i] -= alpha_list[j] * y_hist[j][i]

        # Scale initial Hessian
        if hist_len > 0:
            sy_val = 0.0
            yy_val = 0.0
            for i in range(n):
                sy_val += s_hist[hist_len - 1][i] * y_hist[hist_len - 1][i]
                yy_val += y_hist[hist_len - 1][i] * y_hist[hist_len - 1][i]
            if yy_val > 0:
                gamma = sy_val / yy_val
                for i in range(n):
                    q[i] *= gamma

        # Forward loop
        for j in range(hist_len):
            dot_val = 0.0
            for i in range(n):
                dot_val += y_hist[j][i] * q[i]
            beta_val = rho_hist[j] * dot_val
            for i in range(n):
                q[i] += (alpha_list[j] - beta_val) * s_hist[j][i]

        # Save old x, g; update x
        step = 0.5
        for i in range(n):
            x_old[i] = x[i]
            g_old[i] = g[i]
            x[i] -= step * q[i]

        # Compute new gradient
        for i in range(n):
            g_new[i] = 2.0 * (x[i] - i)
        for i in range(n - 1):
            g_new[i] += 0.1 * x[i + 1]
            g_new[i + 1] += 0.1 * x[i]

        # Update history
        sy_dot = 0.0
        for i in range(n):
            sy_dot += (x[i] - x_old[i]) * (g_new[i] - g_old[i])

        if fabs(sy_dot) > 1e-30:
            if hist_len < m:
                for i in range(n):
                    s_hist[hist_len][i] = x[i] - x_old[i]
                    y_hist[hist_len][i] = g_new[i] - g_old[i]
                rho_hist[hist_len] = 1.0 / sy_dot
                hist_len += 1
            else:
                # Shift history left
                # Swap pointers: move slot 0 to end, shift others down
                tmp_s = s_hist[0]
                tmp_y = y_hist[0]
                for j in range(m - 1):
                    s_hist[j] = s_hist[j + 1]
                    y_hist[j] = y_hist[j + 1]
                    rho_hist[j] = rho_hist[j + 1]
                s_hist[m - 1] = tmp_s
                y_hist[m - 1] = tmp_y
                for i in range(n):
                    s_hist[m - 1][i] = x[i] - x_old[i]
                    y_hist[m - 1][i] = g_new[i] - g_old[i]
                rho_hist[m - 1] = 1.0 / sy_dot

    # Compute final objective
    obj = 0.0
    for i in range(n):
        diff = x[i] - i
        obj += diff * diff
    for i in range(n - 1):
        obj += 0.1 * x[i] * x[i + 1]

    # Free
    for j in range(m):
        free(s_hist[j])
        free(y_hist[j])
    free(s_hist)
    free(y_hist)
    free(rho_hist)
    free(alpha_list)
    free(x)
    free(g)
    free(q)
    free(x_old)
    free(g_old)
    free(g_new)

    return obj
