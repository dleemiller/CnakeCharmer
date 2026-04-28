import numpy as np


def forward_step(a, b):
    n = b.shape[0]
    for i in range(1, n):
        alpha_i = a[i, i]
        alpha_i_1 = a[i - 1, i - 1]
        beta_i_1 = a[i - 1, i]
        gamma_i = a[i, i - 1]

        a[i, i] = alpha_i - beta_i_1 * (gamma_i / alpha_i_1)
        a[i, i - 1] = 0.0
        b[i] = b[i] - b[i - 1] * (gamma_i / alpha_i_1)
    return a, b


def backward_step(a_reduced, b_reduced):
    n = b_reduced.shape[0]
    x = np.zeros(n, dtype=np.float64)
    x[n - 1] = b_reduced[n - 1] / a_reduced[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        b_i = b_reduced[i]
        beta_i = a_reduced[i, i + 1]
        alpha_i = a_reduced[i, i]
        x[i] = (b_i - beta_i * x[i + 1]) / alpha_i

    return x


def thomas_solver(a, b):
    a_reduced, b_reduced = forward_step(a, b)
    return backward_step(a_reduced, b_reduced)
