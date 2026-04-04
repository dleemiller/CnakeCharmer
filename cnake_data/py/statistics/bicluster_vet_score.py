"""Transposed virtual error for synthetic biclusters.

Keywords: statistics, biclustering, virtual error, matrix
"""

from cnake_data.benchmarks import python_benchmark


def _gen_matrix(n_rows: int, n_cols: int, seed: int) -> list[list[float]]:
    state = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    mat = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            v = ((state % 2001) - 1000) / 97.0
            mat[i][j] = v + 0.01 * i - 0.02 * j
    return mat


@python_benchmark(args=(512, 256, 17))
def bicluster_vet_score(n_rows: int, n_cols: int, seed: int) -> tuple:
    m = _gen_matrix(n_rows, n_cols, seed)

    row_means = [0.0] * n_rows
    col_means = [0.0] * n_cols
    col_stds = [0.0] * n_cols

    for j in range(n_cols):
        s1 = 0.0
        s2 = 0.0
        for i in range(n_rows):
            x = m[i][j]
            s1 += x
            s2 += x * x
            row_means[i] += x
        mu = s1 / n_rows
        var = s2 / n_rows - mu * mu
        if var < 1e-12:
            var = 1e-12
        col_means[j] = mu
        col_stds[j] = var**0.5

    s1 = 0.0
    s2 = 0.0
    for i in range(n_rows):
        row_means[i] /= n_rows
        s1 += row_means[i]
        s2 += row_means[i] * row_means[i]

    mu_rho = s1 / n_rows
    sigma_rho2 = s2 / n_rows - mu_rho * mu_rho
    if sigma_rho2 < 1e-12:
        sigma_rho2 = 1e-12
    sigma_rho = sigma_rho2**0.5

    score = 0.0
    for i in range(n_rows):
        rr = (row_means[i] - mu_rho) / sigma_rho
        for j in range(n_cols):
            cc = (m[i][j] - col_means[j]) / col_stds[j]
            d = cc - rr
            if d < 0:
                d = -d
            score += d

    score /= n_rows * n_cols
    return (score, mu_rho, sigma_rho)
