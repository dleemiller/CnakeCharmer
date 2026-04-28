def trace_csr(shape, row_index, col_index, data):
    """Trace of sparse CSR matrix represented by row pointers/cols/data."""
    n_rows, n_cols = shape
    if n_rows != n_cols:
        raise ValueError(f"matrix shape {shape} is not square.")

    total = 0
    for row in range(n_rows):
        for ptr in range(row_index[row], row_index[row + 1]):
            if col_index[ptr] == row:
                total += data[ptr]
                break
    return total


def trace_dense(shape, data):
    """Trace of dense matrix stored in row-major flat data."""
    n_rows, n_cols = shape
    if n_rows != n_cols:
        raise ValueError(f"matrix shape {shape} is not square.")

    total = 0
    ptr = 0
    stride = n_rows + 1
    for _ in range(n_rows):
        total += data[ptr]
        ptr += stride
    return total
