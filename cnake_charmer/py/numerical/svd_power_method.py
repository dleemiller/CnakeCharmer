"""SVD largest singular value via power iteration.

Keywords: numerical, svd, singular value, power iteration, linear algebra, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(250,))
def svd_power_method(n: int) -> tuple:
    """Compute largest singular value of n x n matrix via power iteration.

    Constructs a deterministic n x n matrix A, then uses power iteration
    on A^T * A to find the largest singular value (sqrt of largest eigenvalue).

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (largest_singular_value, final_vector_first, final_vector_last).
    """
    # Build deterministic matrix A (flat, row-major)
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = math.sin((i + 1) * 0.1) * math.cos((j + 1) * 0.2) + 0.5 / (
                abs(i - j) + 1
            )

    # Power iteration on A^T * A
    # Start with initial vector v = [1, 0, 0, ..., 0] normalized
    v = [0.0] * n
    v[0] = 1.0

    num_iters = 30
    sigma = 0.0

    for _ in range(num_iters):
        # Compute w = A * v
        w = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i * n + j] * v[j]
            w[i] = s

        # Compute u = A^T * w = (A^T A) v
        u = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[j * n + i] * w[j]
            u[i] = s

        # Compute norm of u
        norm = 0.0
        for i in range(n):
            norm += u[i] * u[i]
        norm = math.sqrt(norm)

        if norm < 1e-15:
            break

        # Normalize
        for i in range(n):
            v[i] = u[i] / norm

        sigma = math.sqrt(norm)

    return (sigma, v[0], v[n - 1])
