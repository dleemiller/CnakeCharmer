import math


def cround(x):
    return math.ceil(x - 0.5) if x < 0.0 else math.floor(x + 0.5)


def min_img_vec(x, y, img, periodic=True):
    """Minimum-image displacement vector between x and y."""
    dx = [0.0, 0.0, 0.0]
    for i in range(3):
        dx[i] = x[i] - y[i]
        if periodic:
            dx[i] -= cround(dx[i] / img[i]) * img[i]
    return dx


def same_img(x, y, img):
    """Translate x into same periodic image as y."""
    out = list(x)
    for i in range(3):
        dx = out[i] - y[i]
        out[i] -= cround(dx / img[i]) * img[i]
    return out


def min_img(x, img, periodic=True):
    """Wrap coordinates into base periodic image."""
    out = list(x)
    for i in range(3):
        out[i] -= math.floor(out[i] / img[i]) * img[i]
    return out


def min_img_dist_sq(x, y, img, periodic=True):
    """Squared minimum-image distance."""
    dist = 0.0
    for i in range(3):
        dx = x[i] - y[i]
        if periodic:
            dx -= cround(dx / img[i]) * img[i]
        dist += dx * dx
    return dist


def min_img_dist(x, y, img, periodic=True):
    return math.sqrt(min_img_dist_sq(x, y, img, periodic))


def norm3(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def spec_force_inner_loop(w, basis_out, grad, force, r):
    """Accumulate force and gradient contributions."""
    for i in range(len(w)):
        for j in range(len(r)):
            force[j] = force[j] + w[i] * basis_out[i] * r[j]
            grad[i][j] = basis_out[i] * r[j] + grad[i][j]
    return force, grad
