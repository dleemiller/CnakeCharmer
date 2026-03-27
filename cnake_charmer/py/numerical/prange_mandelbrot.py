"""Parallel Mandelbrot set computation, count points in set.

Keywords: numerical, mandelbrot, fractal, parallel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def prange_mandelbrot(n: int) -> int:
    """Count points in the Mandelbrot set on an n x n grid.

    Grid spans real=[-2, 1], imag=[-1.5, 1.5]. Max 100 iterations.

    Args:
        n: Grid resolution (n x n pixels).

    Returns:
        Number of points in the Mandelbrot set.
    """
    count = 0
    max_iter = 100

    for row in range(n):
        ci = -1.5 + 3.0 * row / n
        for col in range(n):
            cr = -2.0 + 3.0 * col / n
            zr = 0.0
            zi = 0.0
            in_set = 1
            for _k in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    in_set = 0
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
            count += in_set

    return count
