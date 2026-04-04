"""Sparse matrix-vector multiplication using CSR format.

Keywords: sparse matrix, CSR, matrix-vector multiply, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def sparse_matvec(n: int) -> tuple:
    """Build an n x n sparse matrix in CSR format and compute y = A * x.

    Matrix construction: row i has nonzeros at columns j = (i + k*37) % n
    for k in 0..4, with value (i * j + 1) % 100 + 1.
    Vector: x[j] = j * 0.001

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (int(y[0]*1e6), int(y[n//2]*1e6), int(sum(y)*1e3) % (10**9)).
    """
    # Build CSR
    row_ptr = [0] * (n + 1)
    col_idx = []
    values = []

    for i in range(n):
        row_ptr[i] = len(col_idx)
        for k in range(5):
            j = (i + k * 37) % n
            v = (i * j + 1) % 100 + 1
            col_idx.append(j)
            values.append(float(v))
    row_ptr[n] = len(col_idx)

    # Build x
    x = [j * 0.001 for j in range(n)]

    # Multiply y = A * x
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            s += values[idx] * x[col_idx[idx]]
        y[i] = s

    total = sum(y)
    return (int(y[0] * 1e6), int(y[n // 2] * 1e6), round(total * 1e3) % (10**9))
