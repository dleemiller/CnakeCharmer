"""Evaluate a parametric function on a 4D grid (h × x × y × z).

For each parameter h, integrates a function over a uniform (x,y,z) grid
using exp and sin. Inspired by spectral/physical field evaluation kernels.

Keywords: parameter sweep, numerical integration, exp, sin, 4d grid
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(12, 10))
def param_grid_eval(n_h: int, n_xyz: int) -> tuple:
    """Evaluate parametric function over h-grid with (x,y,z) inner loop.

    Args:
        n_h: Number of h (parameter) values in [0.5, 2.5].
        n_xyz: Number of x/y/z points per axis in [-1, 1].

    Returns:
        Tuple of (total_sum, first_val, last_val) from the output array.
    """
    h_min = 0.5
    h_max = 2.5
    x_min = -1.0
    x_max = 1.0

    out = [0.0] * n_h

    inv_nh1 = (h_max - h_min) / (n_h - 1) if n_h > 1 else 0.0
    inv_nxyz1 = (x_max - x_min) / (n_xyz - 1) if n_xyz > 1 else 0.0

    for h_i in range(n_h):
        h = h_min + h_i * inv_nh1
        result = 0.0
        for x_i in range(n_xyz):
            X = x_min + x_i * inv_nxyz1
            for y_i in range(n_xyz):
                Y = x_min + y_i * inv_nxyz1
                for z_i in range(n_xyz):
                    Z = x_min + z_i * inv_nxyz1
                    arg = 2.0 * Z + X + Y * Y - h
                    result += math.exp(-(arg * arg)) * (
                        math.sin(X + Y + 3.0 * Z + h) + (Y + Z + h) * (Y + Z + h)
                    )
        out[h_i] = result

    total = 0.0
    for v in out:
        total += v
    return (total, out[0], out[n_h - 1])
