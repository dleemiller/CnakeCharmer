"""Mini-batch SGD for linear regression."""

from __future__ import annotations


def stochastic_gradient_descent(eta, n_iter, x, y, m):
    n = len(x)
    p = len(x[0]) if n else 0

    beta = [0.0] * p
    y_hat = [0.0] * m
    grad = [0.0] * p

    order = list(range(n))
    order.sort(key=lambda i: (i * 1103515245 + 12345) % (2**31))
    x_shuf = [x[i] for i in order]
    y_shuf = [y[i] for i in order]

    j = 1
    while j < n_iter:
        eta_j = eta / (10.0 ** (3.0 * j / max(1, n_iter - 1)))
        lo = (j - 1) * m
        hi = lo + m
        x_j = x_shuf[lo:hi]
        y_j = y_shuf[lo:hi]

        for i in range(len(x_j)):
            s = 0.0
            for k in range(p):
                s += x_j[i][k] * beta[k]
            y_hat[i] = s

        for k in range(p):
            g = 0.0
            for i in range(len(x_j)):
                g += x_j[i][k] * (y_j[i] - y_hat[i])
            grad[k] = (-2.0 / max(1, len(x_j))) * g
            beta[k] = beta[k] - eta_j * grad[k]
        j += 1

    return beta
