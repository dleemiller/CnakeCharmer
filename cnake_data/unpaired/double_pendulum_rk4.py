import math


class Pen2Sim:
    def __init__(self, th0, th1, w0, w1, M0, M1, L0, L1, G, dt):
        self.th0 = th0
        self.th1 = th1
        self.w0 = w0
        self.w1 = w1
        self.M0 = M0
        self.M1 = M1
        self.L0 = L0
        self.L1 = L1
        self.G = G
        self.dt = dt
        self.t = 0.0

    def _rhs(self, th0, th1, w0, w1):
        M0, M1, L0, L1, G = self.M0, self.M1, self.L0, self.L1, self.G
        th0_dot = w0
        th1_dot = w1

        w0_dot = -(
            4.0 * G * M0 * math.sin(th0)
            + 3.0 * G * M1 * math.sin(th0)
            + G * M1 * math.sin(th0 - 2.0 * th1)
            + L0 * M1 * (w0**2) * math.sin(2.0 * th0 - 2.0 * th1)
            + 4.0 * L1 * M1 * (w1**2) * math.sin(th0 - th1)
        ) / (2.0 * L0 * (4.0 * M0 - M1 * math.cos(th0 - th1) ** 2 + 2.0 * M1))

        w1_dot = (
            -(2.0 * M0 + M1) * (G * math.sin(th1) - L0 * (w0**2) * math.sin(th0 - th1))
            + math.cos(th0 - th1)
            * (
                G * M0 * math.sin(th0)
                + G * M1 * math.sin(th0)
                + L1 * M1 * (w1**2) * math.sin(th0 - th1)
            )
        ) / (L1 * (4.0 * M0 - M1 * math.cos(th0 - th1) ** 2 + 2.0 * M1))

        return th0_dot, th1_dot, w0_dot, w1_dot

    def advance(self):
        dt = self.dt
        k0 = self._rhs(self.th0, self.th1, self.w0, self.w1)
        k1 = self._rhs(
            self.th0 + 0.5 * dt * k0[0],
            self.th1 + 0.5 * dt * k0[1],
            self.w0 + 0.5 * dt * k0[2],
            self.w1 + 0.5 * dt * k0[3],
        )
        k2 = self._rhs(
            self.th0 + 0.5 * dt * k1[0],
            self.th1 + 0.5 * dt * k1[1],
            self.w0 + 0.5 * dt * k1[2],
            self.w1 + 0.5 * dt * k1[3],
        )
        k3 = self._rhs(
            self.th0 + dt * k2[0], self.th1 + dt * k2[1], self.w0 + dt * k2[2], self.w1 + dt * k2[3]
        )

        self.th0 += dt * (k0[0] + 2.0 * (k1[0] + k2[0]) + k3[0]) / 6.0
        self.th1 += dt * (k0[1] + 2.0 * (k1[1] + k2[1]) + k3[1]) / 6.0
        self.w0 += dt * (k0[2] + 2.0 * (k1[2] + k2[2]) + k3[2]) / 6.0
        self.w1 += dt * (k0[3] + 2.0 * (k1[3] + k2[3]) + k3[3]) / 6.0
        self.t += dt

        x0 = self.L0 * math.sin(self.th0)
        x1 = x0 + self.L1 * math.sin(self.th1)
        y0 = -self.L0 * math.cos(self.th0)
        y1 = y0 - self.L1 * math.cos(self.th1)
        return (x0, x1, y0, y1, self.t, self.L0, self.L1, self.th0, self.th1)
