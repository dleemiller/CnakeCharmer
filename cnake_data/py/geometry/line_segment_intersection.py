"""Count intersections among n line segments and track intersection coordinates.

Keywords: geometry, line segment, intersection, cross product, coordinate, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def line_segment_intersection(n: int) -> tuple:
    """Count intersections among n line segments with coordinate tracking.

    Segment i: from (sin-like x, cos-like y) to (shifted endpoint).
    Uses cross-product intersection test and computes actual intersection
    points for the first and last intersections found.

    Args:
        n: Number of line segments.

    Returns:
        Tuple of (count, first_intersection_x, last_intersection_y).
    """
    # Generate deterministic segments with more spread
    ax = [0.0] * n
    ay = [0.0] * n
    bx = [0.0] * n
    by = [0.0] * n
    for i in range(n):
        ax[i] = ((i * 73 + 11) % 997) * 0.1
        ay[i] = ((i * 37 + 23) % 991) * 0.1
        bx[i] = ((i * 53 + 7) % 983) * 0.1
        by[i] = ((i * 97 + 13) % 977) * 0.1

    count = 0
    first_ix = 0.0
    last_iy = 0.0
    for i in range(n):
        ax_i = ax[i]
        ay_i = ay[i]
        dx_i = bx[i] - ax_i
        dy_i = by[i] - ay_i
        for j in range(i + 1, n):
            # Cross product test
            ex = ax[j] - ax_i
            ey = ay[j] - ay_i
            fx = bx[j] - ax_i
            fy = by[j] - ay_i

            d1 = dx_i * ey - dy_i * ex
            d2 = dx_i * fy - dy_i * fx

            if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                continue

            dx_j = bx[j] - ax[j]
            dy_j = by[j] - ay[j]
            gx = ax_i - ax[j]
            gy = ay_i - ay[j]
            hx = bx[i] - ax[j]
            hy = by[i] - ay[j]

            d3 = dx_j * gy - dy_j * gx
            d4 = dx_j * hy - dy_j * hx

            if (d3 > 0 and d4 > 0) or (d3 < 0 and d4 < 0):
                continue

            # Compute intersection point using parameter t
            denom = dx_i * dy_j - dy_i * dx_j
            if denom == 0.0:
                # Parallel or collinear
                count += 1
                continue

            t = ((ax[j] - ax_i) * dy_j - (ay[j] - ay_i) * dx_j) / denom
            ix = ax_i + t * dx_i
            iy = ay_i + t * dy_i

            count += 1
            if count == 1:
                first_ix = ix
            last_iy = iy

    return (count, first_ix, last_iy)
