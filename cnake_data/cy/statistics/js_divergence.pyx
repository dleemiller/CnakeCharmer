# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Jensen-Shannon divergence between probability distributions.

Keywords: kl divergence, js divergence, information theory, entropy, probability, cython
"""

from libc.math cimport log
from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def js_divergence(int n):
    """Compute pairwise JS divergence for n probability distributions.

    Args:
        n: Number of distributions (each of dimension 50).

    Returns:
        Tuple of (total_divergence, max_divergence, count_above_threshold).
    """
    cdef int d = 50
    cdef int i, j, k
    cdef double val, total, p, q, m, js
    cdef double total_div = 0.0
    cdef double max_div = 0.0
    cdef int count_above = 0
    cdef double threshold = 0.1

    # Allocate n x d matrix
    cdef double *dists = <double *>malloc(n * d * sizeof(double))
    if not dists:
        raise MemoryError()

    # Generate n deterministic probability distributions
    for i in range(n):
        total = 0.0
        for j in range(d):
            val = <double>(((i * 7 + j * 13 + 3) % 97) + 1)
            dists[i * d + j] = val
            total += val
        # Normalize
        for j in range(d):
            dists[i * d + j] /= total

    for i in range(n):
        for j in range(i + 1, n):
            js = 0.0
            for k in range(d):
                p = dists[i * d + k]
                q = dists[j * d + k]
                m = 0.5 * (p + q)
                if p > 1e-300:
                    js += p * log(p / m)
                if q > 1e-300:
                    js += q * log(q / m)
            js *= 0.5

            total_div += js
            if js > max_div:
                max_div = js
            if js > threshold:
                count_above += 1

    free(dists)
    return (total_div, max_div, count_above)
