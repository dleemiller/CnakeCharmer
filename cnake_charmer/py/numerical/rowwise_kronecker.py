"""Row-wise Kronecker product of two matrices.

Computes D[i,:] = outer(X[i,:], ZV[i,:]).ravel() for each row i,
then returns aggregate statistics over the result matrix.

Keywords: kronecker, outer product, matrix, linear algebra, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def rowwise_kronecker(n: int) -> tuple:
    """Compute row-wise Kronecker product for n rows.

    Each row i has X[i,:] of dimension d=8 and ZV[i,:] of dimension r=6.
    The output D[i,:] has dimension d*r=48 entries, being the flattened
    outer product of X[i,:] and ZV[i,:].

    Args:
        n: Number of rows to process.

    Returns:
        Tuple of (total_sum, max_value) over all D entries.
    """
    d = 8
    r = 6
    out_cols = d * r  # noqa: F841 — documents output width (48)

    total_sum = 0.0
    max_value = -1e300

    for i in range(n):
        # Deterministic row generation
        for j in range(d):
            x_val = ((i * 7 + j * 13 + 3) % 997) / 100.0 - 5.0
            for k in range(r):
                zv_val = ((i * 11 + k * 17 + 7) % 991) / 100.0 - 5.0
                val = x_val * zv_val
                total_sum += val
                if val > max_value:
                    max_value = val

    return (total_sum, max_value)
