"""Distance from points to ellipse boundary.

Keywords: geometry, ellipse, distance, root finding
"""

from cnake_charmer.benchmarks import python_benchmark


def _robust_length(x0: float, x1: float) -> float:
    ax = x0 if x0 >= 0 else -x0
    ay = x1 if x1 >= 0 else -x1
    if ax > ay:
        return ax * (1.0 + (ay / ax) ** 2) ** 0.5
    return ay * (1.0 + (ax / ay) ** 2) ** 0.5


def _get_root(r0: float, z0: float, z1: float) -> float:
    n0 = r0 * z0
    s0 = z1 - 1.0
    s1 = _robust_length(n0, z1)
    for _ in range(100):
        s = (s0 + s1) * 0.5
        if s in (s0, s1):
            return s
        denom0 = s + r0
        denom1 = s + 1.0
        if -1e-15 < denom1 < 1e-15:
            denom1 = 1e-15 if denom1 >= 0.0 else -1e-15
        ratio0 = n0 / denom0
        ratio1 = z1 / denom1
        g = ratio0 * ratio0 + ratio1 * ratio1 - 1.0
        if g > 0.0:
            s0 = s
        elif g < 0.0:
            s1 = s
        else:
            return s
    return (s0 + s1) * 0.5


def _distance_point_ellipse(a: float, b: float, x: float, y: float) -> float:
    if x < 0:
        x = -x
    if y < 0:
        y = -y

    z0 = x / a
    z1 = y / b
    g = z0 * z0 + z1 * z1 - 1.0
    if g == 0.0:
        return 0.0

    r0 = (a / b) ** 0.5
    s = _get_root(r0, z0, z1)
    x0 = r0 * x / (s + r0)
    denom = s + 1.0
    if -1e-15 < denom < 1e-15:
        denom = 1e-15 if denom >= 0.0 else -1e-15
    y0 = y / denom
    dx = x - x0
    dy = y - y0
    return (dx * dx + dy * dy) ** 0.5


@python_benchmark(args=(6.0, 3.0, 70000, 23))
def ellipse_point_distance_batch(a: float, b: float, n_points: int, seed: int) -> tuple:
    state = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    total = 0.0
    max_d = 0.0
    inside = 0

    for _ in range(n_points):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        x = (((state % 20001) - 10000) / 10000.0) * (1.5 * a)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        y = (((state % 20001) - 10000) / 10000.0) * (1.5 * b)
        if (x * x) / (a * a) + (y * y) / (b * b) <= 1.0:
            inside += 1
        d = _distance_point_ellipse(a, b, x, y)
        total += d
        if d > max_d:
            max_d = d

    return (total, max_d, inside)
