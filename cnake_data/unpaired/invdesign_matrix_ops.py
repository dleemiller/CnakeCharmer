import numpy as np


def matrix_M(eps, b=0.0):
    m = np.zeros((4, 4), dtype=np.double)
    m[0, 2] = eps - b**2
    m[1, 3] = eps
    m[2, 0] = 1.0
    m[3, 1] = 1.0 - (b**2 / eps)
    return m


def eigvalsM(eps, b=0.0):
    n = np.sqrt(eps - b**2)
    return np.array([-n, -n, n, n], dtype=np.double)


def eigvecsM(eps, b=0.0):
    sqrtepsb = np.sqrt(eps - b**2)
    c1 = sqrtepsb
    c2 = eps * sqrtepsb / (b**2 - eps)
    c3 = sqrtepsb / (2 * eps)
    c4 = 0.5 / sqrtepsb
    return np.array(
        [
            [[0, -c1, 0, c1], [c2, 0, -c2, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            [[0, -c3, 0, 0.5], [-c4, 0, 0.5, 0], [0, c3, 0, 0.5], [c4, 0, 0.5, 0]],
        ],
        dtype=np.double,
    )


def propagator_layer(f, eps, d, b=0.0):
    scalar_input = np.isscalar(f)
    freqs = np.atleast_1d(f)
    n = freqs.shape[0]
    res = np.empty((n, 4, 4), dtype=np.complex128)

    w = eigvalsM(eps, b)
    vr = eigvecsM(eps, b)
    for i in range(n):
        ik0d = 2j * np.pi * freqs[i] * d
        res[i, :, :] = np.linalg.multi_dot([vr[0], np.diag(np.exp(ik0d * w)), vr[1]])

    return res[0] if scalar_input else res
