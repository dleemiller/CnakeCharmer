import math


def sphere_overlap_covariance_components(x, y, z, r):
    """Average positive overlap score over all unique sphere pairs."""
    points = len(x)
    if not (len(y) == points and len(z) == points and len(r) == points):
        raise ValueError("x, y, z, r must have the same length")

    total = 0.0
    n = 0

    for i in range(points):
        for j in range(i + 1, points):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            denom = r[i] + r[j]
            if denom == 0.0:
                continue
            d = 1.0 - math.sqrt(dx * dx + dy * dy + dz * dz) / denom
            if d > 0.0:
                total += d
                n += 1

    return total / n if n > 0 else 0.0


def compute_group_overlap_covariance(group_sizes, x, y, z, r):
    """Compute overlap covariance independently for contiguous groups."""
    out = []
    start = 0
    for size in group_sizes:
        end = start + size
        out.append(
            sphere_overlap_covariance_components(
                x[start:end], y[start:end], z[start:end], r[start:end]
            )
        )
        start = end
    return out
