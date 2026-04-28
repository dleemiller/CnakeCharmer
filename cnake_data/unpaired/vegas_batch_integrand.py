import math


def vegas_batch_integrand(points):
    """Evaluate a Gaussian-like integrand for a batch of points."""
    n_points = len(points)
    if n_points == 0:
        return []

    dim = len(points[0])
    norm = 1013.2118364296088 ** (dim / 4.0)
    out = [0.0 for _ in range(n_points)]

    for i in range(n_points):
        dx2 = 0.0
        row = points[i]
        for d in range(dim):
            delta = row[d] - 0.5
            dx2 += delta * delta
        out[i] = math.exp(-100.0 * dx2) * norm

    return out
