"""Compute magnetic field from current-carrying wire segments using Biot-Savart law.

Keywords: physics, magnetic field, Biot-Savart, electromagnetism, wire, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def magnetic_field(n: int) -> tuple:
    """Compute magnetic field at observation points from n wire segments.

    Wire segments form a polygonal loop in the xy-plane with vertices at
    (cos(2*pi*i/n), sin(2*pi*i/n), 0). Current I=1.0 A flows through each
    segment. Observation points are on a line at z=0.5 from x=-2 to x=2.
    Uses Biot-Savart: dB = (mu0/4pi) * I * (dl x r_hat) / r^2.

    Args:
        n: Number of wire segments forming the loop.

    Returns:
        Tuple of (sum_bx, sum_by, sum_bz) total field components over all
        observation points.
    """
    pi = math.pi
    mu0_over_4pi = 1e-7  # mu0/(4*pi) in SI
    current = 1.0
    n_obs = 200
    n_sub = 10  # subdivisions per segment for accuracy

    # Precompute wire segment endpoints
    wx = [math.cos(2.0 * pi * i / n) for i in range(n + 1)]
    wy = [math.sin(2.0 * pi * i / n) for i in range(n + 1)]

    total_bx = 0.0
    total_by = 0.0
    total_bz = 0.0

    for obs in range(n_obs):
        ox = -2.0 + 4.0 * obs / (n_obs - 1)
        oy = 0.0
        oz = 0.5

        bx = 0.0
        by = 0.0
        bz = 0.0

        for seg in range(n):
            # Subdivide each segment
            sx0 = wx[seg]
            sy0 = wy[seg]
            sx1 = wx[seg + 1]
            sy1 = wy[seg + 1]

            for sub in range(n_sub):
                t0 = sub / n_sub
                t1 = (sub + 1) / n_sub
                tmid = 0.5 * (t0 + t1)

                # dl vector
                dlx = (sx1 - sx0) / n_sub
                dly = (sy1 - sy0) / n_sub
                dlz = 0.0

                # midpoint of sub-segment
                mx = sx0 + tmid * (sx1 - sx0)
                my = sy0 + tmid * (sy1 - sy0)
                mz = 0.0

                # r vector from source to observation
                rx = ox - mx
                ry = oy - my
                rz = oz - mz

                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                if r < 1e-15:
                    continue
                r3 = r2 * r

                # dl x r (cross product)
                cx = dly * rz - dlz * ry
                cy = dlz * rx - dlx * rz
                cz = dlx * ry - dly * rx

                coeff = mu0_over_4pi * current / r3
                bx += coeff * cx
                by += coeff * cy
                bz += coeff * cz

        total_bx += bx
        total_by += by
        total_bz += bz

    return (total_bx, total_by, total_bz)
