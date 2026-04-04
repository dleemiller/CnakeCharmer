# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute KL/JS-like divergence metrics on generated probability grids (Cython)."""

from libc.math cimport log

from cnake_data.benchmarks import cython_benchmark


cdef void _accumulate_js_entropy(
    double p_bias,
    double q_bias,
    int rows,
    int cols,
    double eps,
    double *kl_pq,
    double *kl_qp,
) noexcept nogil:
    cdef int i, j
    cdef double row_scale, base, p, q
    cdef double pq = 0.0
    cdef double qp = 0.0

    for i in range(rows):
        row_scale = 1.0 + (i & 7) * 0.03
        for j in range(cols):
            base = ((i * 131 + j * 17 + 29) % 1000) / 1000.0
            p = eps + ((base + p_bias) % 1.0) * row_scale
            q = eps + ((base * 0.73 + q_bias + 0.11) % 1.0) * (2.0 - row_scale * 0.25)
            pq += p * log(p / q)
            qp += q * log(q / p)

    kl_pq[0] = pq
    kl_qp[0] = qp


@cython_benchmark(syntax="cy", args=(0.03, 0.07, 220, 240, 1e-12))
def js_entropy_grid(double p_bias, double q_bias, int rows, int cols, double eps):
    cdef double kl_pq = 0.0
    cdef double kl_qp = 0.0

    with nogil:
        _accumulate_js_entropy(p_bias, q_bias, rows, cols, eps, &kl_pq, &kl_qp)

    return (kl_pq, kl_qp, 0.5 * (kl_pq + kl_qp))
