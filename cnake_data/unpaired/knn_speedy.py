import numpy as np
import numpy.linalg as la


class Kernel:
    def __init__(self, ktype="euclidean", sigma=1):
        self.sigma = sigma
        self.ktype = ktype
        self.f = None
        if ktype in ("euclidean", "minkowski"):
            self.f = self.euclid_fast
        if ktype == "cosine":
            self.f = self.cosine
        if ktype == "gaussian":
            self.f = self.gaussian
        if ktype == "poly2":
            self.f = self.poly2

    def euclid(self, xi, xj):
        return np.sqrt(np.sum([(xi[m] - xj[m]) ** 2 for m in range(xi.shape[0])]))

    def euclid_fast(self, x_test, x_train, i, j):
        result = 0.0
        m = x_test.shape[1]
        for k in range(m):
            result += (x_test[i, k] - x_train[j, k]) ** 2
        return np.sqrt(result)

    def cosine(self, x, xt, i, j):
        return 1 - (np.dot(x[i], xt[j].T) / (la.norm(x[i]) * la.norm(xt[j])))

    def gaussian(self, x, xt, i, j, sigma):
        return np.sum(
            [
                -np.sqrt(la.norm(a - b) ** 2 / (2 * sigma**2))
                for a, b in zip(x[i], xt[j], strict=False)
            ]
        )

    def poly2(self, x, xt, i, j):
        return np.dot(x[i], xt[j]) ** 2


def calc_K(kernel, x_test, x_train):
    n_samples = x_test.shape[0]
    n_samples_train = x_train.shape[0]
    k = np.zeros((n_samples, n_samples_train), dtype=float)
    for i in range(n_samples):
        for j in range(n_samples_train):
            k[i, j] = kernel.f(x_test, x_train, i, j)
    return k
