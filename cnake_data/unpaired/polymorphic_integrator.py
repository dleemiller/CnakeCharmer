class Integrand:
    def f(self, x):
        raise NotImplementedError()


class QuarticMinusLinear(Integrand):
    def f(self, x):
        return x * x * x * x - 3.0 * x


def integrate_f(integrand, a, b, n):
    """Riemann-sum integration with polymorphic integrand.f(x)."""
    s = 0.0
    dx = (b - a) / n
    for i in range(n):
        s += integrand.f(a + i * dx)
    return s * dx
