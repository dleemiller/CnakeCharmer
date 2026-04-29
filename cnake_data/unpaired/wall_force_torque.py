"""Wall interaction force and torque for a 2D rod-like particle."""

from __future__ import annotations

import math


class WallForce:
    def __init__(self, length, bond_l, gravity):
        self.r_cut = 3.0 ** (1.0 / 6.0)
        self.length = float(length)
        self.bond_l = float(bond_l)
        self.gravity = float(gravity)

    def wall_f(self, x):
        if x < self.r_cut:
            r = x
            sign = 1.0
        elif x > self.length - self.r_cut:
            r = self.length - x
            sign = -1.0
        else:
            return 0.0
        return sign * 3.0 * (3.0 * r**-10 - r**-4)

    def force(self, x, y, theta):
        c_th = math.cos(theta)
        s_th = math.sin(theta)
        x1 = x + c_th * self.bond_l
        x2 = x - c_th * self.bond_l
        f1 = self.wall_f(x1)
        f2 = self.wall_f(x2)
        fx = f1 + f2 - self.gravity
        fy = 0.0
        torque = s_th * self.bond_l / 2.0 * (f2 - f1)
        return fx, fy, torque
