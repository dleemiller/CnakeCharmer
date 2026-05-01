"""Ideal-gas primitive/conservative conversion and sound speed."""

from __future__ import annotations

import numpy as np


class IdealGas:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def conservative_from_primitive(self, density, pressure, volume, velocity):
        density = np.asarray(density, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        volume = np.asarray(volume, dtype=float)
        velocity = np.asarray(velocity, dtype=float)

        mass = density * volume
        momentum = velocity * mass[:, None]
        v_sq = np.sum(velocity * velocity, axis=1)
        energy = (0.5 * density * v_sq + pressure / (self.gamma - 1.0)) * volume
        return mass, momentum, energy

    def primitive_from_conservative(self, mass, momentum, energy, volume):
        mass = np.asarray(mass, dtype=float)
        momentum = np.asarray(momentum, dtype=float)
        energy = np.asarray(energy, dtype=float)
        volume = np.asarray(volume, dtype=float)

        density = mass / volume
        velocity = momentum / mass[:, None]
        v_sq = np.sum(velocity * velocity, axis=1)
        pressure = (energy / volume - 0.5 * density * v_sq) * (self.gamma - 1.0)
        return density, velocity, pressure

    def sound_speed(self, density, pressure):
        return np.sqrt(self.gamma * pressure / density)

    def get_gamma(self):
        return self.gamma
