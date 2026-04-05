# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute state probabilities in an M/M/c/K/m-style queue model (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 9ef6f7df00318562de207b973704905ad0a83496
- filename: erlang.pyx
"""

from cnake_data.benchmarks import cython_benchmark


cdef double _power_faculty(double x, int n) noexcept nogil:
    cdef double res = 1.0
    cdef int i
    for i in range(1, n + 1):
        res = res * x / i
    return res


cdef double _cn(double lam, double mu, double nu, int agents, int n) noexcept nogil:
    cdef double res
    cdef int i
    cdef double div
    if n <= agents:
        return _power_faculty(lam / mu, n)
    res = _power_faculty(lam / mu, agents)
    for i in range(1, n - agents + 1):
        div = agents * mu + i * nu
        if div != 0.0:
            res *= lam / div
    return res


cdef void _mmckm_kernel(
    double lam,
    double mu,
    double nu,
    int agents,
    int capacity,
    int* p0_out,
    int* pmid_out,
    int* plast_out,
    int* bestn_out,
) noexcept nogil:
    cdef double norm = 0.0
    cdef double p0, p_mid, p_last, pn
    cdef double best_p = -1.0
    cdef int best_n = 0
    cdef int n, mid

    for n in range(capacity + 1):
        norm += _cn(lam, mu, nu, agents, n)

    if norm > 0.0:
        p0 = 1.0 / norm
    else:
        p0 = 0.0

    mid = capacity // 2
    p_mid = _cn(lam, mu, nu, agents, mid) * p0
    p_last = _cn(lam, mu, nu, agents, capacity) * p0

    for n in range(capacity + 1):
        pn = _cn(lam, mu, nu, agents, n) * p0
        if pn > best_p:
            best_p = pn
            best_n = n

    p0_out[0] = <int>(p0 * 1000000)
    pmid_out[0] = <int>(p_mid * 1000000)
    plast_out[0] = <int>(p_last * 1000000)
    bestn_out[0] = best_n


@cython_benchmark(syntax="cy", args=(55, 31, 11, 12, 700))
def stack_mmckm_probability(int lam_num, int mu_num, int nu_num, int agents, int capacity):
    cdef double lam = lam_num / 10.0
    cdef double mu = mu_num / 10.0
    cdef double nu = nu_num / 20.0
    cdef int p0i = 0
    cdef int pmidi = 0
    cdef int plasti = 0
    cdef int best_n = 0

    with nogil:
        _mmckm_kernel(lam, mu, nu, agents, capacity, &p0i, &pmidi, &plasti, &best_n)

    return (p0i, pmidi, plasti, best_n)
