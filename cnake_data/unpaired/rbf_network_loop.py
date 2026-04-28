import math

import numpy as np


def rbf_network(x, beta, theta):
    x = np.asarray(x, dtype=float)
    beta = np.asarray(beta, dtype=float)

    n = x.shape[0]
    d = x.shape[1]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            r = 0.0
            for k in range(d):
                r += (x[j, k] - x[i, k]) ** 2
            r = math.sqrt(r)
            y[i] += beta[j] * math.exp(-((r * theta) ** 2))

    return y
