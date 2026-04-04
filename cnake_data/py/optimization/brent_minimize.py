"""Brent's method for function minimization.

Keywords: brent, minimization, golden section, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def brent_minimize(n: int) -> float:
    """Find minima of f_i(x) = x^4 - 3x^2 + x + i*0.001 using Brent's method.

    Tests n different offset functions on [-3, 3].
    Returns sum of all minima found.

    Args:
        n: Number of function variants to minimize.

    Returns:
        Sum of minimum values found.
    """
    golden = 0.3819660112501051  # (3 - sqrt(5)) / 2
    total = 0.0

    for idx in range(n):
        offset = idx * 0.001
        ax = -3.0
        bx = 0.0
        cx = 3.0

        # Brent's method
        a = ax
        b = cx
        if a > b:
            a, b = b, a

        x = bx
        w = bx
        v = bx
        fx = x * x * x * x - 3.0 * x * x + x + offset
        fw = fx
        fv = fx
        e = 0.0
        d = 0.0

        for _ in range(50):
            midpoint = 0.5 * (a + b)
            tol1 = 1e-8 * abs(x) + 1e-10
            tol2 = 2.0 * tol1

            if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                break

            # Try parabolic interpolation
            if abs(e) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                else:
                    q = -q
                if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                    e = d
                    d = p / q
                else:
                    e = b - x if x < midpoint else a - x
                    d = golden * e
            else:
                e = b - x if x < midpoint else a - x
                d = golden * e

            u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)

            fu = u * u * u * u - 3.0 * u * u + u + offset

            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v in (x, w):
                    v = u
                    fv = fu

        total += fx

    return total
