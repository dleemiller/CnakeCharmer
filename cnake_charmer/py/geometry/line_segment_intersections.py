"""
Count intersections among n line segments using brute force.

Keywords: geometry, line segment, intersection, cross product, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def line_segment_intersections(n: int) -> int:
    """Count the number of intersecting pairs among n line segments.

    Segment i goes from (i*0.7, i*1.3) to ((i*3+1)%n * 0.7, (i*3+1)%n * 1.3).
    Uses brute-force O(n^2) with cross-product-based intersection test.

    Args:
        n: Number of line segments.

    Returns:
        Tuple of (number of intersecting segment pairs, last_i, last_j indices).
    """
    # Precompute segment endpoints
    ax = [0.0] * n
    ay = [0.0] * n
    bx = [0.0] * n
    by = [0.0] * n
    for i in range(n):
        ax[i] = i * 0.7
        ay[i] = i * 1.3
        j = (i * 3 + 1) % n
        bx[i] = j * 0.7
        by[i] = j * 1.3

    count = 0
    last_i = -1
    last_j = -1
    for i in range(n):
        ax_i = ax[i]
        ay_i = ay[i]
        dx_i = bx[i] - ax_i
        dy_i = by[i] - ay_i
        for j in range(i + 1, n):
            # Check if segments i and j intersect using cross products
            # Vector from A_i to A_j and A_i to B_j
            ex = ax[j] - ax_i
            ey = ay[j] - ay_i
            fx = bx[j] - ax_i
            fy = by[j] - ay_i

            d1 = dx_i * ey - dy_i * ex
            d2 = dx_i * fy - dy_i * fx

            if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                continue

            # Vector from A_j to A_i and A_j to B_i
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

            count += 1
            last_i = i
            last_j = j

    return (count, last_i, last_j)
