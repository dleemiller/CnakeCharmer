import math


class Archimedian:
    """Archimedian spiral point generator."""

    def start(self, param):
        self.param = float(param)
        self.x = 0.0
        self.y = 0.0
        self.step_size = 0.5
        self.u = 0
        self.v = 0
        self.r = 0.0

    def get_next(self):
        self.r += self.step_size
        self.x = self.param * self.r * math.cos(self.r)
        self.y = self.param * self.r * math.sin(self.r)

        if int(self.x - self.u) == 0 and int(self.y - self.v) == 0:
            return self.get_next()

        self.u = int(self.x)
        self.v = int(self.y)
        return self.u, self.v


class Rectangular:
    """Rectangular spiral coordinate generator."""

    def start(self, param):
        self.param = int(param)
        self.u = 0
        self.v = 0
        self.phase = 0
        self.step = 0
        self.m = self.param

    def get_next(self):
        if self.phase == 0:
            if self.step < self.m:
                self.v -= 1
                self.step += 1
            else:
                self.step = 0
                self.phase = 1
                return self.get_next()

        elif self.phase == 1:
            if self.step < self.m:
                self.u += 1
                self.step += 1
            else:
                self.step = 0
                self.phase = 2
                self.m += self.param
                return self.get_next()

        elif self.phase == 2:
            if self.step < self.m:
                self.v += 1
                self.step += 1
            else:
                self.step = 0
                self.phase = 3
                return self.get_next()

        elif self.phase == 3:
            if self.step < self.m:
                self.u -= 1
                self.step += 1
            else:
                self.step = 0
                self.phase = 0
                self.m += self.param
                return self.get_next()

        return self.u, self.v
