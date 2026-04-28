def canova_dp(p, x, degree, indices, n_nz):
    """Compute ANOVA DP table over sparse feature indices."""
    n_features = len(p)
    a = [[0.0 for _ in range(n_features + 1)] for _ in range(degree + 1)]
    a[0] = [1.0 for _ in range(n_features + 1)]

    for t in range(1, degree + 1):
        j_prev = t - 2
        for jj in range(t - 1, n_nz):
            j = indices[jj]
            a[t][j + 1] = a[t][j_prev + 1] + p[j] * x[jj] * a[t - 1][j_prev + 1]
            j_prev = j

    return a


def canova_saving_memory(p, x, degree, indices, n_nz):
    """Memory-reduced ANOVA DP variant."""
    a = [0.0 for _ in range(degree + 1)]
    a[0] = 1.0

    for jj in range(min(degree, n_nz)):
        j = indices[jj]
        for t in range(jj, -1, -1):
            a[t + 1] += a[t] * p[j] * x[jj]

    for jj in range(degree, n_nz):
        j = indices[jj]
        for t in range(degree - 1, -1, -1):
            a[t + 1] += a[t] * p[j] * x[jj]

    return a


def cgrad_anova_coordinate_wise(p_js, x_iij, degree, a):
    """Coordinate-wise gradient recurrence for ANOVA terms."""
    grad = [0.0 for _ in range(degree + 1)]
    grad[1] = x_iij
    for t in range(2, degree + 1):
        grad[t] = x_iij * (a[t - 1] - p_js * grad[t - 1])
    return grad
