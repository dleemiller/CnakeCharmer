import math


def _robust_length(x0, x1):
    """Compute robust Euclidean length avoiding overflow."""
    if x0 > x1:
        return x0 * math.sqrt(1.0 + (x1 / x0) ** 2)
    else:
        return x1 * math.sqrt(1.0 + (x0 / x1) ** 2)


def _get_root(r0, z0, z1, g):
    """Find root via bisection for ellipse distance computation."""
    max_iterations = 100
    i = 0
    n0 = r0 * z0
    s0 = z1 - 1
    s1 = _robust_length(n0, z1)
    s = 0.0

    while i < max_iterations:
        s = (s0 + s1) / 2.0
        if s in (s0, s1):
            return s
        ratio0 = n0 / (s + r0)
        ratio1 = z1 / (s + 1)
        g = ratio0**2 + ratio1**2 - 1
        if g > 0.0:
            s0 = s
        elif g < 0.0:
            s1 = s
        else:
            return s
        i += 1

    return 1.0e30


def distance_point_ellipse(a, b, x, y):
    """Compute the shortest distance from a point to an ellipse.

    Uses the algorithm from "Distance from a Point to an Ellipse, an Ellipsoid,
    or a Hyperellipsoid" by David Eberly (Geometric Tools).

    Parameters
    ----------
    a : float
        Semi-major axis length of the ellipse.
    b : float
        Semi-minor axis length of the ellipse.
    x : float
        X-coordinate of the query point (must be >= 0).
    y : float
        Y-coordinate of the query point (must be >= 0).

    Returns
    -------
    float
        The shortest distance from point (x, y) to the ellipse defined by
        (X/a)^2 + (Y/b)^2 = 1, considering only the first quadrant.
    """
    if y > 0:
        if x > 0:
            z0 = x / a
            z1 = y / b
            g = z0**2 + z1**2 - 1
            if g != 0:
                r0 = math.sqrt(a / b)
                sbar = _get_root(r0, z0, z1, g)
                x0 = r0 * x / (sbar + r0)
                y0 = y / (sbar + 1)
                dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            else:
                dist = 0.0
        else:
            dist = abs(y - b)
    else:
        numer0 = a * x
        denom0 = math.sqrt(a) - math.sqrt(b)
        if numer0 < denom0:
            xde0 = numer0 / denom0
            x0 = a * xde0
            y0 = b * math.sqrt(1 - xde0**2)
            dist = math.sqrt((x0 - x) ** 2 + y0**2)
        else:
            dist = abs(x - a)
    return dist
