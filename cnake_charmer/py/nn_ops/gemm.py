"""General matrix multiply (GEMM) — full C = A * B.

Simulates a fully-connected layer forward pass in a neural network.
All three tiers compute the full product matrix, then extract trace
to return a scalar for verification.

Keywords: matrix multiply, gemm, linear algebra, neural network, blas, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def gemm(n: int) -> float:
    """Compute full C = A * B for n×n matrices, return trace(C).

    Args:
        n: Matrix dimension.

    Returns:
        Trace of the product matrix.
    """
    # Allocate matrices
    A = [[(i + j) % 100 / 10.0 for j in range(n)] for i in range(n)]
    B = [[(i - j + n) % 100 / 10.0 for j in range(n)] for i in range(n)]
    C = [[0.0] * n for _ in range(n)]

    # Full matrix multiply
    for i in range(n):
        for k in range(n):
            a_ik = A[i][k]
            for j in range(n):
                C[i][j] += a_ik * B[k][j]

    # Extract trace
    trace = 0.0
    for i in range(n):
        trace += C[i][i]

    return trace
