def rectangle_fill_grid(n):
    """Fill an n x n grid by painting colored rectangles and summing pixel values.

    Generates n deterministic rectangles with coordinates and RGB colors,
    paints them onto the grid by accumulating color values, then computes
    statistics over the result.

    Args:
        n: Grid dimension and number of rectangles to paint.

    Returns:
        (total_r, total_g, total_b, max_pixel_sum) across the grid.
    """
    grid_r = [[0] * n for _ in range(n)]
    grid_g = [[0] * n for _ in range(n)]
    grid_b = [[0] * n for _ in range(n)]

    for row in range(n):
        x0 = (row * 3 + 1) % n
        y0 = (row * 7 + 2) % n
        x1 = min(x0 + (row % 5) + 1, n)
        y1 = min(y0 + (row % 4) + 1, n)
        r = (row * 17 + 5) % 256
        g = (row * 31 + 11) % 256
        b = (row * 53 + 23) % 256

        for i in range(y0, y1):
            for j in range(x0, x1):
                grid_r[i][j] += r
                grid_g[i][j] += g
                grid_b[i][j] += b

    total_r = 0
    total_g = 0
    total_b = 0
    max_pixel_sum = 0

    for i in range(n):
        for j in range(n):
            total_r += grid_r[i][j]
            total_g += grid_g[i][j]
            total_b += grid_b[i][j]
            pixel_sum = grid_r[i][j] + grid_g[i][j] + grid_b[i][j]
            if pixel_sum > max_pixel_sum:
                max_pixel_sum = pixel_sum

    return (total_r, total_g, total_b, max_pixel_sum)
