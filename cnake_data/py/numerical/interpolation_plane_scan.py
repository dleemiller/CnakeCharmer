"""Scan planar interpolation residual statistics over a rectangular grid.

Keywords: interpolation, planar model, residual scan, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(0.17, -0.05, 1.3, -64, 64, -48, 48, 4, 0.73))
def interpolation_plane_scan(
    a: float,
    b: float,
    c: float,
    x_start: int,
    x_stop: int,
    y_start: int,
    y_stop: int,
    passes: int,
    blend: float,
) -> tuple:
    """Accumulate interpolation and residual metrics over a rectangular scan."""
    sum_interp = 0.0
    weighted_error = 0.0
    max_abs_error = 0.0

    for p in range(passes):
        bias = (p + 1) * 0.03125
        for y in range(y_start, y_stop):
            for x in range(x_start, x_stop):
                z = a * x + b * y + c
                observed = z * blend + (x - y) * 0.25 + bias
                err = observed - z
                abs_err = err if err >= 0.0 else -err
                sum_interp += z
                weighted_error += abs_err * (1.0 + ((x + y + p) & 3) * 0.125)
                if abs_err > max_abs_error:
                    max_abs_error = abs_err

    return (sum_interp, weighted_error, max_abs_error)
