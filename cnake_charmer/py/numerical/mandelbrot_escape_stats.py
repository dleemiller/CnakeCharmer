"""
Mandelbrot membership/escape statistics on a regular grid.

Sourced from SFT DuckDB blob: 6b0c1ba7b17cea70784b6fa6f3d31831b357ba46
Keywords: mandelbrot, fractal, escape time, numerical
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(350, 220, 64))
def mandelbrot_escape_stats(width: int, height: int, max_iter: int) -> tuple:
    """Return (inside_count, escape_iter_sum, edge_inside_count)."""
    min_x = -2.0
    min_y = -1.2
    dx = 3.0 / width
    dy = 2.4 / height

    inside = 0
    edge_inside = 0
    escape_sum = 0

    for ix in range(width):
        real = min_x + ix * dx
        for iy in range(height):
            imag = min_y + iy * dy
            zr = 0.0
            zi = 0.0
            escaped_at = 0
            for it in range(1, max_iter + 1):
                zr, zi = zr * zr - zi * zi + real, 2.0 * zr * zi + imag
                if zr * zr + zi * zi >= 4.0:
                    escaped_at = it
                    break

            if escaped_at == 0:
                inside += 1
                if ix == 0 or iy == 0 or ix == width - 1 or iy == height - 1:
                    edge_inside += 1
            else:
                escape_sum += escaped_at

    return (inside, escape_sum, edge_inside)
