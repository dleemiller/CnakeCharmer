"""Monin-Obukhov stability length batch computation.

Iteratively solves for the Obukhov stability length L from the bulk
Richardson number Rib for a range of atmospheric stability conditions.
Used in atmospheric boundary layer modelling.

Keywords: Monin-Obukhov, atmospheric boundary layer, stability length, Richardson number, physics
"""

import math

from cnake_data.benchmarks import python_benchmark


def _psim(zeta: float) -> float:
    """Integrated stability correction for momentum."""
    if zeta <= 0.0:
        # Use sqrt(sqrt(x)) instead of x**0.25 for identical float ops
        tmp = math.sqrt(1.0 - 16.0 * zeta)
        x = math.sqrt(tmp)
        return (
            math.pi / 2.0
            - 2.0 * math.atan(x)
            + math.log((1.0 + x) * (1.0 + x) * (1.0 + x * x) / 8.0)
        )
    return -2.0 / 3.0 * (zeta - 5.0 / 0.35) * math.exp(-0.35 * zeta) - zeta - (10.0 / 3.0) / 0.35


def _psih(zeta: float) -> float:
    """Integrated stability correction for heat."""
    if zeta <= 0.0:
        tmp = math.sqrt(1.0 - 16.0 * zeta)
        x = math.sqrt(tmp)
        return 2.0 * math.log((1.0 + x * x) / 2.0)
    # Use v * sqrt(v) instead of v**1.5 for identical float ops
    v = 1.0 + (2.0 / 3.0) * zeta
    return (
        -2.0 / 3.0 * (zeta - 5.0 / 0.35) * math.exp(-0.35 * zeta)
        - v * math.sqrt(v)
        - (10.0 / 3.0) / 0.35
        + 1.0
    )


def _ribtol(rib: float, zsl: float, z0m: float, z0h: float) -> float:
    """Iteratively find Obukhov length L for given bulk Richardson number."""
    L = 1.0 if rib > 0.0 else -1.0
    L0 = 2.0 * L

    while abs(L - L0) > 0.001:
        L0 = L
        ds = math.log(zsl / z0m) - _psim(zsl / L) + _psim(z0m / L)
        fx = rib - zsl / L * (math.log(zsl / z0h) - _psih(zsl / L) + _psih(z0h / L)) / (ds * ds)

        Ls = L - 0.001 * L
        Le = L + 0.001 * L
        ds = math.log(zsl / z0m) - _psim(zsl / Ls) + _psim(z0m / Ls)
        de = math.log(zsl / z0m) - _psim(zsl / Le) + _psim(z0m / Le)
        fxs = -zsl / Ls * (math.log(zsl / z0h) - _psih(zsl / Ls) + _psih(z0h / Ls)) / (ds * ds)
        fxe = -zsl / Le * (math.log(zsl / z0h) - _psih(zsl / Le) + _psih(z0h / Le)) / (de * de)
        fxdif = (fxe - fxs) / (0.002 * L)

        if fxdif != 0.0:
            L = L - fx / fxdif
        else:
            break
        if L > 1e4:
            L = 1e4
        elif L < -1e4:
            L = -1e4

    return L


@python_benchmark(args=(500,))
def monin_obukhov_stability(n: int) -> tuple:
    """Compute Obukhov length L for n bulk Richardson numbers in [-0.5, 0.5].

    Args:
        n: Number of Rib values to process.

    Returns:
        Tuple of (mean_L, first_L, last_L).
    """
    zsl = 10.0
    z0m = 0.1
    z0h = 0.01

    total = 0.0
    first = 0.0
    last = 0.0

    for i in range(n):
        rib = -0.5 + i * (1.0 / (n - 1)) if n > 1 else 0.0
        if abs(rib) < 0.01:
            rib = 0.01
        L = _ribtol(rib, zsl, z0m, z0h)
        total += L
        if i == 0:
            first = L
        last = L

    return (total / n, first, last)
