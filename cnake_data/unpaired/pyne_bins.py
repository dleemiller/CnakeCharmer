import numpy as np
from numpy import logspace


def ninespace(start, stop, num=50, endpoint=True):
    log_start = np.log10(1.0 - start)
    log_stop = np.log10(1.0 - stop)
    samples = 1.0 - logspace(log_start, log_stop, num, endpoint)
    return samples


def stair_step(x, y):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    g = len(y)
    assert g + 1 == len(x)

    xss = np.empty(2 * g, dtype=x.dtype)
    yss = np.empty(2 * g, dtype=y.dtype)
    xss[:-1:2] = x[:-1]
    xss[1::2] = x[1:]
    yss[::2] = y
    yss[1::2] = y
    return xss, yss


def pointwise_linear_collapse(x_g, x, y):
    return pointwise_collapse(x_g, x, y)


def pointwise_collapse(x_g, x, y, logx=False, logy=False, log=False):
    x_g = np.asarray(x_g, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    g = x_g.shape[0] - 1
    n = x.shape[0] - 1

    if not (np.all(np.diff(x) >= 0.0) or np.all(np.diff(x) <= 0.0)):
        raise ValueError("x must be monotonically increasing/decreasing.")
    if not (np.all(np.diff(x_g) >= 0.0) or np.all(np.diff(x_g) <= 0.0)):
        raise ValueError("x_g must be monotonically increasing/decreasing.")

    reversed_order = False
    if x_g[0] > x_g[-1]:
        x_g = x_g[::-1]
        x = x[::-1]
        y = y[::-1]
        reversed_order = True

    if logx or log:
        if x_g[0] <= 0.0 or x[0] <= 0.0:
            raise ValueError("x values must be positive for logrithmic interpolation")
        x_g = np.log(x_g)
        x = np.log(x)
    if logy or log:
        if y[0] <= 0.0:
            raise ValueError("y values must be positive for logrithmic interpolation")
        y = np.log(y)

    n0 = 0
    n1 = 1
    y_g = np.empty(g, dtype="float64")

    for g0 in range(g):
        g1 = g0 + 1
        val = 0.0
        while n1 < n and x[n1] <= x_g[g1]:
            if x_g[g0] <= x[n0]:
                val += 0.5 * (y[n1] + y[n0]) * (x[n1] - x[n0])
            else:
                ylower = ((y[n1] - y[n0]) / (x[n1] - x[n0])) * (x_g[g0] - x[n0]) + y[n0]
                val += 0.5 * (y[n1] + ylower) * (x[n1] - x_g[g0])
            n0 += 1
            n1 += 1

        yupper = ((y[n1] - y[n0]) / (x[n1] - x[n0])) * (x_g[g1] - x[n0]) + y[n0]
        if x_g[g0] <= x[n0]:
            val += 0.5 * (yupper + y[n0]) * (x_g[g1] - x[n0])
        else:
            ylower = ((y[n1] - y[n0]) / (x[n1] - x[n0])) * (x_g[g0] - x[n0]) + y[n0]
            val += 0.5 * (yupper + ylower) * (x_g[g1] - x_g[g0])
        y_g[g0] = val / (x_g[g1] - x_g[g0])

    if reversed_order:
        y_g = y_g[::-1]
    if logy or log:
        y_g = np.exp(y_g)
    return y_g
