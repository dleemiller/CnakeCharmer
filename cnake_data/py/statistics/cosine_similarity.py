"""
Cosine similarity between pairs of deterministic vectors.

Computes cosine similarity = dot(a,b) / (norm(a) * norm(b)) for n vector pairs,
returning the sum of all similarities plus the last similarity as a discriminator.

Keywords: statistics, cosine, similarity, dot product, vector, norm, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def cosine_similarity(n: int) -> tuple:
    """Compute cosine similarity for n pairs of 64-dimensional vectors.

    Vectors are deterministically generated: a[j] = sin-like hash, b[j] = cos-like hash.
    Returns (sum_of_similarities, last_similarity).

    Args:
        n: Number of vector pairs to process.

    Returns:
        Tuple of (sum of cosine similarities, last cosine similarity).
    """
    dim = 64
    total = 0.0
    last = 0.0

    for i in range(n):
        # Build deterministic vectors
        dot_ab = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for j in range(dim):
            # Deterministic pseudo-random values in [-1, 1]
            seed_a = ((i * 1013 + j * 7 + 3) % 9973) / 4986.5 - 1.0
            seed_b = ((i * 2017 + j * 13 + 11) % 9967) / 4983.5 - 1.0
            dot_ab += seed_a * seed_b
            norm_a += seed_a * seed_a
            norm_b += seed_b * seed_b

        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        sim = dot_ab / denom if denom > 0.0 else 0.0

        total += sim
        last = sim

    return (total, last)
