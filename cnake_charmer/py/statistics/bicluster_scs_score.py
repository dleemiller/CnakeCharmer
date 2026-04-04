"""Submatrix correlation score for synthetic biclusters.

Keywords: statistics, biclustering, correlation, pearson
"""

from cnake_charmer.benchmarks import python_benchmark


def _gen_matrix(n_rows: int, n_cols: int, seed: int) -> list[list[float]]:
    state = (seed * 1103515245 + 12345) & 0xFFFFFFFF
    mat = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            mat[i][j] = ((state % 4096) - 2048) / 157.0 + 0.03 * i + 0.02 * j
    return mat


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    sx = sy = sx2 = sy2 = sxy = 0.0
    for i in range(n):
        xv = x[i]
        yv = y[i]
        sx += xv
        sy += yv
        sx2 += xv * xv
        sy2 += yv * yv
        sxy += xv * yv
    num = n * sxy - sx * sy
    den1 = n * sx2 - sx * sx
    den2 = n * sy2 - sy * sy
    if den1 <= 1e-12 or den2 <= 1e-12:
        return 0.0
    return num / ((den1 * den2) ** 0.5)


@python_benchmark(args=(72, 56, 19))
def bicluster_scs_score(n_rows: int, n_cols: int, seed: int) -> tuple:
    m = _gen_matrix(n_rows, n_cols, seed)

    min_row_score = 2.0
    for i in range(n_rows):
        t = 0.0
        for i2 in range(n_rows):
            if i != i2:
                v = _pearson(m[i], m[i2])
                if v < 0:
                    v = -v
                t += v
        row_score = 1.0 - t / (n_rows - 1)
        if row_score < min_row_score:
            min_row_score = row_score

    cols = [[m[i][j] for i in range(n_rows)] for j in range(n_cols)]
    min_col_score = 2.0
    for j in range(n_cols):
        t = 0.0
        for j2 in range(n_cols):
            if j != j2:
                v = _pearson(cols[j], cols[j2])
                if v < 0:
                    v = -v
                t += v
        col_score = 1.0 - t / (n_cols - 1)
        if col_score < min_col_score:
            min_col_score = col_score

    return (min(min_row_score, min_col_score), min_row_score, min_col_score)
