"""Compute escape statistics over a Mandelbrot tile.

Keywords: numerical, mandelbrot, fractal, escape time, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(-2.0, 1.0, 380, 120, 4.0))
def mandelbrot_tile_stats(
    min_val: float, max_val: float, size: int, max_iter: int, threshold: float
) -> tuple:
    """Return aggregate escape metrics over a square grid in the complex plane."""
    step = (abs(min_val) + max_val) / (size - 1)
    sum_escape = 0.0
    stable_count = 0
    edge_checksum = 0.0

    for i in range(size):
        imag = max_val - step * i
        for j in range(size):
            real = min_val + step * j
            zr = 0.0
            zi = 0.0
            escaped = max_iter
            for it in range(max_iter):
                zr2 = zr * zr - zi * zi + real
                zi2 = 2.0 * zr * zi + imag
                zr = zr2
                zi = zi2
                if zr * zr + zi * zi > threshold:
                    escaped = it + 1
                    break
            sum_escape += escaped
            if escaped == max_iter:
                stable_count += 1
            if i == 0 or j == 0 or i == size - 1 or j == size - 1:
                edge_checksum += escaped * (1.0 + ((i + j) & 1) * 0.25)

    return (sum_escape, float(stable_count), edge_checksum)
