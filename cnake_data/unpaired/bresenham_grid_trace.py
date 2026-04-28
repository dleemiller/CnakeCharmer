def grid_increment(x0, y0, x1, y1, mult, grid):
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if x0 == x1 and y0 == y1:
            break

        if 0 <= x0 < nrows and 0 <= y0 < ncols:
            grid[x0][y0] += mult

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def bresenham(points, grid):
    x1 = y1 = 0
    for k in range(len(points) - 1):
        x1, y1 = points[k + 1]
        x0, y0 = points[k]
        grid_increment(x0, y0, x1, y1, 1, grid)

    if grid and 0 <= x1 < len(grid) and 0 <= y1 < len(grid[0]):
        grid[x1][y1] += 1


def bresenham_multiply(points, mult, grid):
    x1 = y1 = 0
    k = 0
    for k in range(len(points) - 1):
        x1, y1 = points[k + 1]
        x0, y0 = points[k]
        grid_increment(x0, y0, x1, y1, mult[k], grid)

    if grid and len(mult) and 0 <= x1 < len(grid) and 0 <= y1 < len(grid[0]):
        grid[x1][y1] += mult[k]
