class OnlineMoments:
    """Online mean/variance accumulator using Welford-style updates."""

    def __init__(self):
        self.mu = 0.0
        self.v = 0.0
        self.n = 0

    def append(self, x):
        mu_prev = self.mu
        self.n += 1
        self.mu = mu_prev + (x - mu_prev) / self.n
        self.v = self.v + (x - self.mu) * (x - mu_prev)

    def mean_variance(self):
        if self.n == 0:
            return 0.0, 0.0
        return self.mu, self.v / self.n


class LinearFriction1D:
    def __init__(self, gamma):
        self.gamma = gamma

    def force(self, x):
        return -self.gamma * x


class LinearFrictionND:
    def __init__(self, gamma):
        self.gamma = gamma

    def force(self, x):
        return [-self.gamma * xi for xi in x]
