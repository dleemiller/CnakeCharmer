"""Gaussian and Gaussian-mixture evaluation kernels."""

from __future__ import annotations

import math

SQRT2PI = math.sqrt(2.0 * math.pi)


def gaussian(outbuf, x, mu, sigma):
    two_sigma2 = 2.0 * (sigma * sigma)
    norm = SQRT2PI * sigma
    for i in range(len(outbuf)):
        x_less_mu = x[i] - mu
        outbuf[i] += math.exp(-(x_less_mu * x_less_mu) / two_sigma2) / norm


def gaussians(outbuf, x, mus, sigmas):
    for i in range(len(x)):
        acc = 0.0
        for k in range(len(mus)):
            sigma = sigmas[k]
            two_sigma2 = 2.0 * (sigma * sigma)
            norm = SQRT2PI * sigma
            x_less_mu = x[i] - mus[k]
            acc += math.exp(-(x_less_mu * x_less_mu) / two_sigma2) / norm
        outbuf[i] += acc
