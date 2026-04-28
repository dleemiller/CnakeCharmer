def calc_mutation_matrix_entries(d, u, v):
    """Build COO triplets for 1D forward/backward mutation operator."""
    data = []
    row = []
    col = []

    for i in range(d):
        if i > 0:
            data.extend([u * (d - i), -v * i])
            row.extend([i, i])
            col.extend([i - 1, i])
        if i < d - 1:
            data.extend([-u * (d - i - 1), v * (i + 1)])
            row.extend([i, i])
            col.extend([i, i + 1])

    return data, row, col


def calc_drift_matrix_entries(d):
    """Build COO triplets for 1D drift operator."""
    data = []
    row = []
    col = []

    for i in range(d):
        if i > 1:
            data.append((i - 1) * (d - i))
            row.append(i)
            col.append(i - 1)
        if i < d - 2:
            data.append((i + 1) * (d - i - 2))
            row.append(i)
            col.append(i + 1)
        if 0 < i < d - 1:
            data.append(-2 * i * (d - i - 1))
            row.append(i)
            col.append(i)

    return data, row, col


def calc_drift_matrix_dense(d):
    """Dense drift matrix equivalent for easier inspection/testing."""
    res = [[0.0 for _ in range(d)] for _ in range(d)]

    for i in range(d):
        if i > 1:
            res[i][i - 1] = float((i - 1) * (d - i))
        if i < d - 2:
            res[i][i + 1] = float((i + 1) * (d - i - 2))
        if 0 < i < d - 1:
            res[i][i] = float(-2 * i * (d - i - 1))

    return res
