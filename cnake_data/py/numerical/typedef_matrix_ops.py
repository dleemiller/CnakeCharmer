"""4x4 matrix multiplication using typedef arrays.

Keywords: numerical, matrix, multiplication, linear algebra, typedef, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def typedef_matrix_ops(n: int) -> float:
    """Repeatedly multiply 4x4 matrices, return trace of result.

    Initializes two 4x4 matrices deterministically, then multiplies
    them n times (accumulating into result). Returns the trace.

    Args:
        n: Number of matrix multiplications to perform.

    Returns:
        Trace of the final 4x4 result matrix.
    """
    # Matrix A: a[i][j] = ((i * 4 + j) * 7 + 3) % 17 / 10.0
    # Matrix B: b[i][j] = ((i * 4 + j) * 11 + 5) % 19 / 10.0
    a = [0.0] * 16
    b = [0.0] * 16
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            a[idx] = ((i * 4 + j) * 7 + 3) % 17 / 10.0
            b[idx] = ((i * 4 + j) * 11 + 5) % 19 / 10.0

    # Result starts as identity
    result = [0.0] * 16
    for i in range(4):
        result[i * 4 + i] = 1.0

    for _ in range(n):
        # result = result * a + scale * b (to prevent overflow/underflow)
        temp = [0.0] * 16
        for i in range(4):
            for j in range(4):
                s = 0.0
                for k in range(4):
                    s += result[i * 4 + k] * a[k * 4 + j]
                temp[i * 4 + j] = s
        # Add small fraction of b to keep values bounded
        for i in range(16):
            result[i] = temp[i] * 0.1 + b[i] * 0.01

    trace = 0.0
    for i in range(4):
        trace += result[i * 4 + i]

    return trace
