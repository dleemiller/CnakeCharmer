"""General matrix multiply (GEMM) with trace computation.

Keywords: matrix multiply, gemm, linear algebra, neural network, blas
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def gemm(n: int) -> float:
    """Compute C = A * B for n x n matrices and return trace(C).

    A[i][j] = (i + j) % 100 / 10.0
    B[i][j] = (i - j + n) % 100 / 10.0

    Args:
        n: Matrix dimension.

    Returns:
        Trace of C as float.
    """
    trace = 0.0
    for i in range(n):
        s = 0.0
        for k in range(n):
            a_ik = ((i + k) % 100) / 10.0
            b_ki = ((k - i + n) % 100) / 10.0
            s += a_ik * b_ki
        trace += s
    return trace
