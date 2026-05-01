"""Optimizer update rules including Nesterov and Adam variants."""

from __future__ import annotations

import numpy as np


def linear_decay(v0: float, decay: float, t: int) -> float:
    return v0 / (1.0 + decay * t)


class Optimizer:
    def __init__(self, alpha=0.001, b1=0.9, b2=0.999, eps=1e-8, b1_decay=0.0, b2_decay=0.0, L2=0.0):
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.b1_decay = b1_decay
        self.b2_decay = b2_decay
        self.L2 = L2
        self.nr_update = {}
        self.mom1 = {}
        self.mom2 = {}

    def __call__(self, weights, gradient, lr_scale=1.0, key="default"):
        self.nr_update[key] = self.nr_update.get(key, 0) + 1
        nr_upd = self.nr_update[key]
        if self.L2 != 0:
            gradient = gradient + self.L2 * weights
        if self.b1 > 0.0 and self.b2 > 0.0:
            self._adam(weights, gradient, lr_scale, key, nr_upd)
        elif self.b1 > 0.0:
            self._nesterov(weights, gradient, lr_scale, key)
        else:
            weights -= lr_scale * self.alpha * gradient
        return weights

    def _nesterov(self, weights, gradient, lr_scale, key):
        lr = self.alpha * lr_scale
        momentum = self.mom1.setdefault(key, np.zeros_like(weights))
        weights += -self.b1 * momentum
        momentum *= self.b1
        momentum -= lr * gradient
        weights += (1 + self.b1) * momentum

    def _adam(self, weights, gradient, lr_scale, key, nr_upd):
        mom1 = self.mom1.setdefault(key, np.zeros_like(weights))
        mom2 = self.mom2.setdefault(key, np.zeros_like(weights))
        lr = self.alpha * lr_scale
        b1 = linear_decay(self.b1, self.b1_decay, nr_upd)
        b2 = linear_decay(self.b2, self.b2_decay, nr_upd)
        mom1[:] = b1 * mom1 + (1.0 - b1) * gradient
        mom2[:] = b2 * mom2 + (1.0 - b2) * (gradient * gradient)
        mhat = mom1 / (1.0 - b1)
        vhat = mom2 / (1.0 - b2)
        weights -= lr * mhat / (np.sqrt(vhat) + self.eps)
