import numpy as np


def percent_error(fofy, y):
    fofy = np.asarray(fofy, dtype=float)
    y = np.asarray(y, dtype=float)
    eps = (float(np.max(y)) + 0.0001) / 1000.0
    dm = np.abs((fofy / (eps + y)) - 1.0)
    return float(np.mean(dm))


def percent_error_2d(fofy, y):
    fofy = np.asarray(fofy, dtype=float)
    y = np.asarray(y, dtype=float)
    total = 0.0
    for j in range(fofy.shape[0]):
        total += percent_error(fofy[j], y[j])
    return float(total)


def least_squares(fofy, y, w):
    fofy = np.asarray(fofy, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    dm = fofy - y
    return float(np.sum(dm * dm * w))


def least_squares_2d(fofy, y, w):
    fofy = np.asarray(fofy, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    dm = fofy - y
    return float(np.sum(dm * dm * w))
