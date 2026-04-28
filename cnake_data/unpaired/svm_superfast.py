import numpy as np


def inner_2d(x, i, j):
    result = 0.0
    for k in range(x.shape[1]):
        result += x[i, k] * x[j, k]
    return result


class Kernel:
    def __init__(self, ktype="linear", degree=3, coef0=0.0):
        self.ktype = ktype
        self.degree = degree
        self.coef0 = coef0

    def f_fast(self, x, i, j):
        if self.ktype == "linear":
            return inner_2d(x, i, j)
        if self.ktype == "poly":
            return self.f(x[i], x[j])
        raise NotImplementedError(self.ktype)

    def f(self, x, y):
        if self.ktype == "linear":
            return np.inner(x, y)
        if self.ktype == "poly":
            return (self.coef0 + np.inner(x, y)) ** self.degree
        raise NotImplementedError(self.ktype)


def _bias(b, ei, yi, yj, aidiff, ajdiff, i, j, k):
    return b - ei - yi * aidiff * k[i, i] - yj * ajdiff * k[i, j]


def _eval_f(idx, x, y, alphas, bias, k):
    sigma = bias
    for i in range(x.shape[0]):
        sigma += alphas[i] * y[i] * k[idx, i]
    return sigma


def my_lagrangian(x, y, kernel, c=1.0, tolerance=1e-5, maxiter=100):
    b = 0.0
    alphas = np.zeros(x.shape[0], dtype=float)

    k = np.zeros((x.shape[0], x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k[i, j] = kernel.f_fast(x, i, j)

    passes = 0.0
    while passes < maxiter:
        n_changed = 0.0
        for i in range(x.shape[0]):
            fx = _eval_f(i, x, y, alphas, b, k)
            ei = fx - y[i]
            ye = y[i] * ei
            if (ye < -tolerance and alphas[i] < c) or (ye > tolerance and alphas[i] > 0):
                j = np.random.randint(0, x.shape[0] - 1)
                if j >= i:
                    j += 1
                ej = _eval_f(j, x, y, alphas, b, k) - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] != y[j]:
                    l = max(0.0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0.0, alphas[i] + alphas[j] - c)
                    h = min(c, alphas[i] + alphas[j])

                if l == h:
                    continue

                rho = 2.0 * k[i, j] - k[i, i] - k[j, j]
                if rho >= 0:
                    continue

                alphas[j] = alpha_j_old - y[j] * (ei - ej) / rho
                alphas[j] = min(max(alphas[j], l), h)
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])

                ajdiff = alphas[j] - alpha_j_old
                aidiff = alphas[i] - alpha_i_old
                b1 = _bias(b, ei, y[i], y[j], aidiff, ajdiff, i, j, k)
                b2 = _bias(b, ej, y[j], y[i], ajdiff, aidiff, j, i, k)
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                n_changed += 1.0

        passes = passes + 1.0 if n_changed == 0 else 0.0

    return alphas, b
