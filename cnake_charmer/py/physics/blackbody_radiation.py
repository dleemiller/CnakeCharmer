"""Compute Planck spectral radiance at multiple wavelengths.

Keywords: physics, blackbody, planck, radiation, spectral, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def blackbody_radiation(n: int) -> float:
    """Compute Planck spectral radiance at n wavelengths for T=5778K.

    lambda[i] = (0.1 + i*0.01) * 1e-6 m.
    B = 2*h*c^2 / lambda^5 * 1/(exp(h*c/(lambda*k*T)) - 1).
    Returns sum of spectral radiances.

    Args:
        n: Number of wavelength samples.

    Returns:
        Sum of spectral radiances as a float.
    """
    h = 6.626e-34
    c = 2.998e8
    kb = 1.381e-23
    T = 5778.0
    two_hc2 = 2.0 * h * c * c
    hc_over_kT = h * c / (kb * T)

    total = 0.0
    for i in range(n):
        lam = (0.1 + i * 0.01) * 1e-6
        lam2 = lam * lam
        lam5 = lam2 * lam2 * lam
        exponent = hc_over_kT / lam
        if exponent > 700.0:
            continue
        B = two_hc2 / lam5 / (math.exp(exponent) - 1.0)
        total += B

    return total
