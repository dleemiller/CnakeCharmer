"""Sparse matrix-vector multiply in CSR format.

Keywords: sparse matrix, CSR, matrix-vector multiply, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def sparse_matrix_vector(n: int) -> float:
    """Sparse matrix-vector multiply in CSR format.

    Matrix is n x n with 3 nonzeros per row at columns
    (i*3+1)%n, (i*7+2)%n, (i*11+3)%n with values 1.0, 2.0, 3.0.
    Vector: v[i] = i * 0.1.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (sum of result vector, result[0], result[n//2]).
    """
    # Build CSR arrays
    nnz = 3 * n
    row_ptr = [0] * (n + 1)
    col_idx = [0] * nnz
    values = [0.0] * nnz

    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + 3
        base = i * 3
        col_idx[base] = (i * 3 + 1) % n
        col_idx[base + 1] = (i * 7 + 2) % n
        col_idx[base + 2] = (i * 11 + 3) % n
        values[base] = 1.0
        values[base + 1] = 2.0
        values[base + 2] = 3.0

    # Vector
    vec = [i * 0.1 for i in range(n)]

    # SpMV
    result = [0.0] * n
    total = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(row_ptr[i], row_ptr[i + 1]):
            row_sum += values[j] * vec[col_idx[j]]
        result[i] = row_sum
        total += row_sum

    result_at_0 = result[0]
    result_at_half = result[n // 2]
    return (total, result_at_0, result_at_half)
