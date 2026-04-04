"""Rasterize rotated ellipses and rectangles on an integer grid.

Keywords: image_processing, raster, ellipse, rectangle, coverage
"""

from cnake_charmer.benchmarks import python_benchmark


def _sin_t(theta: float) -> float:
    x = theta
    x2 = x * x
    return x - x * x2 / 6.0 + x * x2 * x2 / 120.0


def _cos_t(theta: float) -> float:
    x2 = theta * theta
    return 1.0 - x2 / 2.0 + x2 * x2 / 24.0


@python_benchmark(args=(48, 36, 48, 17))
def raster_shape_coverage(width: int, height: int, n_shapes: int, seed: int) -> tuple:
    grid = [0] * (width * height)
    state = (seed * 1103515245 + 12345) & 0xFFFFFFFF

    for s in range(n_shapes):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        cx = int(state % width)
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        cy = int(state % height)
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        sx = 2 + int(state % (max(3, width // 5)))
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        sy = 2 + int(state % (max(3, height // 5)))
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        ang = ((state % 6283) / 1000.0) - 3.1415
        ca = _cos_t(ang)
        sa = _sin_t(ang)

        for y in range(height):
            for x in range(width):
                rx = ca * (x - cx) - sa * (y - cy)
                ry = sa * (x - cx) + ca * (y - cy)
                hit = False
                if (s & 1) == 0:
                    hit = (rx * rx) / (sx * sx) + (ry * ry) / (sy * sy) <= 1.0
                else:
                    hit = (-sx <= rx <= sx) and (-sy <= ry <= sy)
                if hit:
                    grid[y * width + x] += 1

    covered = 0
    overlap = 0
    checksum = 0
    for i, v in enumerate(grid):
        if v > 0:
            covered += 1
        if v > 1:
            overlap += 1
        checksum = (checksum + v * (i + 1)) & 0xFFFFFFFF

    return (covered, overlap, checksum)
