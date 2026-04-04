"""
Mandelbrot set point counting.

Keywords: numerical, mandelbrot, fractal, complex numbers, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def mandelbrot_count(n: int) -> int:
    """Count points in an n x n grid that belong to the Mandelbrot set.

    Grid covers [-2, 1] x [-1.5, 1.5] with max 100 iterations.

    Args:
        n: Grid resolution (n x n points).

    Returns:
        Tuple of (number of points in the Mandelbrot set, total iteration sum).
    """
    count = 0
    iteration_sum = 0
    max_iter = 100

    for row in range(n):
        for col in range(n):
            cr = -2.0 + col * 3.0 / (n - 1)
            ci = -1.5 + row * 3.0 / (n - 1)
            zr = 0.0
            zi = 0.0
            iteration = 0
            while iteration < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                iteration += 1
            iteration_sum += iteration
            if iteration == max_iter:
                count += 1

    return (count, iteration_sum)
