import math


def matern_kernel(x, nu=2.5):
    """Approximate Matern-family kernel for selected smoothness values."""
    if x == 0.0:
        return 1.0
    if nu == 0.5:
        return math.exp(-x)
    if nu == 1.5:
        s3 = math.sqrt(3.0)
        return (1.0 + s3 * x) * math.exp(-s3 * x)
    if nu == 2.5:
        s5 = math.sqrt(5.0)
        return (1.0 + s5 * x + (5.0 / 3.0) * x * x) * math.exp(-s5 * x)

    # Fallback to Gaussian-like limit for large nu.
    return math.exp(-0.5 * x * x)


def exponential_kernel(x):
    """Exponential kernel (Matern with nu=0.5)."""
    return matern_kernel(x, nu=0.5)


def square_exponential_kernel(x):
    """Squared-exponential/Gaussian kernel."""
    return math.exp(-0.5 * x * x)


def rational_quadratic_kernel(x, alpha=1.0):
    """Rational-quadratic kernel."""
    base = 1.0 + (x * x) / (2.0 * alpha)
    return base ** (-alpha) if alpha != 1.0 else 1.0 / base


def euclidean_distance(point1, point2, scale=1.0):
    """Scaled Euclidean distance between vectors."""
    d2 = 0.0
    for a, b in zip(point1, point2, strict=False):
        diff = a - b
        d2 += diff * diff
    return math.sqrt(d2) / scale
