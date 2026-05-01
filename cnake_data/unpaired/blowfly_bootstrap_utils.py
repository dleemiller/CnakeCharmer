"""Bootstrap transition/probability helper functions."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln


def func(x, particle, ancestor, delta, shape):
    coeff = np.exp(
        gammaln(ancestor + 1)
        - gammaln(ancestor - particle + 1)
        - gammaln(particle + 1)
        + shape * np.log(shape)
        - gammaln(shape)
    )
    return (
        coeff
        * np.exp(-(delta * particle + shape) * x)
        * ((1 - np.exp(-delta * x)) ** (ancestor - particle))
        * (x ** (shape - 1))
    )


def transi_s(particles, ancestor, delta, shape, dim):
    output = np.empty(dim)
    for i in range(dim):
        particle = particles[i]
        inte = quad(func, 0, np.inf, args=(particle, ancestor, delta, shape))[0]
        output[i] = inte
    return output


def coeff_r(particle, shape, dim):
    output = np.empty(dim)
    for i in range(dim):
        output[i] = np.exp(gammaln(particle[i] + shape) - gammaln(particle[i] + 1) - gammaln(shape))
    return output


def coeff_beta(ancestor, p, n0, dim):
    output = np.empty(dim)
    for i in range(dim):
        output[i] = p * ancestor * np.exp(-ancestor / n0)
    return output


def proba_r(coeff, shape, dim):
    output = np.empty(dim)
    for i in range(dim):
        output[i] = shape / (coeff[i] + shape)
    return output


def calc_delta(rs, particles, next_obs, dim, tol):
    output = np.empty(dim, dtype=np.int32)
    for i in range(dim):
        diff = rs[i] + particles[i] - next_obs
        if (0 <= diff <= tol * next_obs) or (0 <= -diff <= tol * next_obs):
            output[i] = 1
        else:
            output[i] = 0
    return output
