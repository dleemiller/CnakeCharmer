import math

import numpy as np


def integrand(x, params):
    q0, qs, z_hat, radius, eta, qp, qpp, p, l = params
    return (
        -l * x
        + (q0 + p + l * qs) * z_hat / radius * math.cos(x)
        + qp / (eta * radius) * z_hat * (1.0 - math.cos(x))
        - qpp * qs * z_hat * z_hat / (4.0 * eta * eta * radius * radius) * math.sin(x) * math.cos(x)
    )


def hl2_real(x, params):
    return math.cos(integrand(x, params)) / (2.0 * math.pi)


def hl2_imag(x, params):
    return math.sin(integrand(x, params)) / (2.0 * math.pi)


def _integrate_func(func, params, n_grid=4096):
    xs = np.linspace(0.0, 2.0 * math.pi, n_grid)
    ys = np.array([func(x, params) for x in xs], dtype=float)
    return float(np.trapz(ys, xs))


def hlp2_serial(q0, qs, z_hat, radius, eta, qp, qpp, l, p_max):
    results = np.empty(2 * p_max + 1, dtype=float)
    for pp in range(-p_max, p_max + 1):
        params = (q0, qs, z_hat, radius, eta, qp, qpp, pp, l)
        ore = _integrate_func(hl2_real, params)
        oim = _integrate_func(hl2_imag, params)
        results[pp + p_max] = ore * ore + oim * oim
    return results


def hlp2_parallel(q0, qs, z_hat, radius, eta, qp, qpp, l, p_max=50000):
    # Python version keeps same API; computes serially.
    return hlp2_serial(q0, qs, z_hat, radius, eta, qp, qpp, l, p_max)
