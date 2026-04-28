"""3D Gaussian field with last-point cache."""

from __future__ import annotations

from math import exp


class GaussianVolume:
    def __init__(self, peak: float, sigma: float):
        self.peak = float(peak)
        self.sigma = float(sigma)
        self._constant = 2.0 * self.sigma * self.sigma

        self._cache_valid = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    def evaluate(self, x: float, y: float, z: float) -> float:
        if self._cache_valid and x == self._cache_x and y == self._cache_y and z == self._cache_z:
            return self._cache_v

        v = self.peak * exp(-(x * x + y * y + z * z) / self._constant)
        self._cache_valid = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = v
        return v
