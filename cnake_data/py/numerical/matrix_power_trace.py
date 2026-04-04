"""Compute trace of a matrix raised to a power.

Keywords: matrix, power, trace, linear algebra, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(150,))
def matrix_power_trace(n: int) -> float:
    """Compute trace(A^4) for an n x n matrix using repeated matrix multiplication.

    Generates a deterministic matrix A, computes A^4 = A*A*A*A, and returns
    the trace (sum of diagonal elements).

    Args:
        n: Matrix dimension (n x n).

    Returns:
        Trace of A^4.
    """
    # Generate matrix A (flat layout, row-major)
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            h = ((i * 2654435761 + j * 1103515245) >> 12) & 0xFFF
            A[i * n + j] = (h % 201 - 100) / 100.0

    def mat_mul(X, Y, size):
        R = [0.0] * (size * size)
        for i in range(size):
            for k in range(size):
                x_ik = X[i * size + k]
                if x_ik == 0.0:
                    continue
                for j in range(size):
                    R[i * size + j] += x_ik * Y[k * size + j]
            pass
        return R

    # Compute A^4 = ((A*A)*(A*A))
    A2 = mat_mul(A, A, n)
    A4 = mat_mul(A2, A2, n)

    # Trace
    trace = 0.0
    for i in range(n):
        trace += A4[i * n + i]

    return trace
