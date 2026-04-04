"""Fanno flow compressible gas dynamics.

Keywords: fanno flow, compressible flow, gas dynamics, mach number, thermodynamics
"""

import math

from cnake_data.benchmarks import python_benchmark


def _p_pstar(ma, gamma):
    """Pressure ratio p/p* for Fanno flow."""
    return (1.0 / ma) * ((gamma + 1) / 2) ** 0.5 * (1 + (gamma - 1) / 2 * ma * ma) ** (-0.5)


def _t_tstar(ma, gamma):
    """Temperature ratio T/T* for Fanno flow."""
    return (gamma + 1) / (2 * (1 + (gamma - 1) / 2 * ma * ma))


def _rho_rhostar(ma, gamma):
    """Density ratio rho/rho* for Fanno flow."""
    return (1.0 / ma) * ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * ma * ma)) ** 0.5


def _nondim_length(ma, gamma):
    """Non-dimensional length parameter 4fL*/D for Fanno flow."""
    m2 = ma * ma
    gp1 = gamma + 1
    gm1 = gamma - 1
    return (1 - m2) / (gamma * m2) + gp1 / (2 * gamma) * math.log(
        gp1 * m2 / (2 * (1 + gm1 / 2 * m2))
    )


def _fanno_ma_from_length(l_param, gamma, ma0=0.1):
    """Find Mach number from length parameter using secant method."""
    x1 = ma0
    x2 = ma0 + 0.01
    target = l_param

    for _ in range(100):
        f1 = _nondim_length(x1, gamma) - target
        f2 = _nondim_length(x2, gamma) - target
        if abs(f2) < 1e-12:
            break
        if abs(f2 - f1) < 1e-15:
            break
        x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        x1 = x2
        x2 = x3

    return x2


@python_benchmark(args=(2000,))
def fanno_flow(n):
    """Compute Fanno flow properties for n Mach numbers.

    Args:
        n: Number of Mach number evaluation points.

    Returns:
        Tuple of (total_pressure_ratio, total_temp_ratio, total_length_param).
    """
    gamma = 1.4
    total_p = 0.0
    total_t = 0.0
    total_l = 0.0

    for i in range(n):
        ma = 0.05 + (i * 2.95) / (n - 1) if n > 1 else 1.0

        p_ratio = _p_pstar(ma, gamma)
        t_ratio = _t_tstar(ma, gamma)
        r_ratio = _rho_rhostar(ma, gamma)

        total_p += p_ratio
        total_t += t_ratio * r_ratio

        if ma < 0.99 or ma > 1.01:
            l_param = _nondim_length(ma, gamma)
            # Verify inversion
            ma_inv = _fanno_ma_from_length(l_param, gamma, 0.1 if ma < 1.0 else 1.5)
            total_l += abs(ma - ma_inv)

    return (total_p, total_t, total_l)
