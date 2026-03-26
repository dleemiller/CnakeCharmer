"""Nelder-Mead simplex optimization.

Keywords: nelder-mead, simplex, optimization, direct search, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def simplex_nelder_mead(n: int) -> float:
    """Nelder-Mead optimization of f(x,y) = (x-1)^2 + (y-2)^2 + sin(x*y).

    Tests n different starting points. 100 iterations each.
    Returns sum of final f values.

    Args:
        n: Number of optimization runs.

    Returns:
        Sum of final objective values.
    """
    total = 0.0

    for idx in range(n):
        x0 = (idx * 0.7) % 5 - 2.5
        y0 = (idx * 1.3) % 5 - 2.5

        # Initialize simplex: 3 vertices for 2D
        # v0 = (x0, y0), v1 = (x0+1, y0), v2 = (x0, y0+1)
        vx0 = x0
        vy0 = y0
        vx1 = x0 + 1.0
        vy1 = y0
        vx2 = x0
        vy2 = y0 + 1.0

        f0 = (vx0 - 1.0) ** 2 + (vy0 - 2.0) ** 2 + math.sin(vx0 * vy0)
        f1 = (vx1 - 1.0) ** 2 + (vy1 - 2.0) ** 2 + math.sin(vx1 * vy1)
        f2 = (vx2 - 1.0) ** 2 + (vy2 - 2.0) ** 2 + math.sin(vx2 * vy2)

        for _ in range(100):
            # Sort: f_best <= f_mid <= f_worst
            if f0 > f1:
                vx0, vx1 = vx1, vx0
                vy0, vy1 = vy1, vy0
                f0, f1 = f1, f0
            if f0 > f2:
                vx0, vx2 = vx2, vx0
                vy0, vy2 = vy2, vy0
                f0, f2 = f2, f0
            if f1 > f2:
                vx1, vx2 = vx2, vx1
                vy1, vy2 = vy2, vy1
                f1, f2 = f2, f1

            # Centroid of best two
            cx = (vx0 + vx1) * 0.5
            cy = (vy0 + vy1) * 0.5

            # Reflection
            rx = 2.0 * cx - vx2
            ry = 2.0 * cy - vy2
            fr = (rx - 1.0) ** 2 + (ry - 2.0) ** 2 + math.sin(rx * ry)

            if fr < f1:
                if fr < f0:
                    # Expansion
                    ex = 3.0 * cx - 2.0 * vx2
                    ey = 3.0 * cy - 2.0 * vy2
                    fe = (ex - 1.0) ** 2 + (ey - 2.0) ** 2 + math.sin(ex * ey)
                    if fe < fr:
                        vx2 = ex
                        vy2 = ey
                        f2 = fe
                    else:
                        vx2 = rx
                        vy2 = ry
                        f2 = fr
                else:
                    vx2 = rx
                    vy2 = ry
                    f2 = fr
            else:
                # Contraction
                if fr < f2:
                    kx = 0.5 * (cx + rx)
                    ky = 0.5 * (cy + ry)
                else:
                    kx = 0.5 * (cx + vx2)
                    ky = 0.5 * (cy + vy2)
                fk = (kx - 1.0) ** 2 + (ky - 2.0) ** 2 + math.sin(kx * ky)

                if fk < f2:
                    vx2 = kx
                    vy2 = ky
                    f2 = fk
                else:
                    # Shrink
                    vx1 = vx0 + 0.5 * (vx1 - vx0)
                    vy1 = vy0 + 0.5 * (vy1 - vy0)
                    vx2 = vx0 + 0.5 * (vx2 - vx0)
                    vy2 = vy0 + 0.5 * (vy2 - vy0)
                    f1 = (vx1 - 1.0) ** 2 + (vy1 - 2.0) ** 2 + math.sin(vx1 * vy1)
                    f2 = (vx2 - 1.0) ** 2 + (vy2 - 2.0) ** 2 + math.sin(vx2 * vy2)

        # Best value
        best = f0
        if f1 < best:
            best = f1
        if f2 < best:
            best = f2
        total += best

    return total
