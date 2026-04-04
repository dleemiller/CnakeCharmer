# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cosine similarity between pairs of deterministic vectors (Cython-optimized).

Keywords: statistics, cosine, similarity, dot product, vector, norm, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def cosine_similarity(int n):
    """Compute cosine similarity for n pairs of 64-dimensional vectors.

    Args:
        n: Number of vector pairs to process.

    Returns:
        Tuple of (sum of cosine similarities, last cosine similarity).
    """
    cdef int dim = 64
    cdef double total = 0.0
    cdef double last = 0.0
    cdef double dot_ab, norm_a, norm_b, seed_a, seed_b, denom, sim
    cdef int i, j

    for i in range(n):
        dot_ab = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for j in range(dim):
            seed_a = ((i * 1013 + j * 7 + 3) % 9973) / 4986.5 - 1.0
            seed_b = ((i * 2017 + j * 13 + 11) % 9967) / 4983.5 - 1.0
            dot_ab += seed_a * seed_b
            norm_a += seed_a * seed_a
            norm_b += seed_b * seed_b

        denom = sqrt(norm_a) * sqrt(norm_b)
        if denom > 0.0:
            sim = dot_ab / denom
        else:
            sim = 0.0

        total += sim
        last = sim

    return (total, last)
