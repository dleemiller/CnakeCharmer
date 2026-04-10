import math


def compute_pi(n):
    """Approximate pi using numerical integration of sqrt(1 - x^2).

    Divides [-1, 1] into n intervals and sums the semicircle area.
    Returns the approximation of pi.
    """
    delta = 1.0 / n
    total_sum = 0.0
    for i in range(n):
        x = -1.0 + i * delta
        total_sum += math.sqrt(1.0 - x * x)
    return 4.0 * total_sum / n
