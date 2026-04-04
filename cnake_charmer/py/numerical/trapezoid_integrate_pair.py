"""
Compare trapezoid and midpoint integration on a synthetic function.

Sourced from SFT DuckDB blob: 2600a4533970a0cc22f44b551f7c008fdb0294a2
Keywords: trapezoid rule, midpoint rule, integration, numerical
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(0.0, 6.0, 200000, 0.7))
def trapezoid_integrate_pair(a: float, b: float, steps: int, freq: float) -> tuple:
    """Integrate f(x)=x*x+0.5*x+sin-like polynomial proxy at two quadratures."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    dx = (b - a) / steps
    trap = 0.0
    mid = 0.0

    for i in range(steps):
        x0 = a + i * dx
        x1 = x0 + dx
        xm = x0 + 0.5 * dx
        f0 = x0 * x0 + 0.5 * x0 + freq * x0 * (1.0 - x0 * 0.01)
        f1 = x1 * x1 + 0.5 * x1 + freq * x1 * (1.0 - x1 * 0.01)
        fm = xm * xm + 0.5 * xm + freq * xm * (1.0 - xm * 0.01)
        trap += 0.5 * (f0 + f1) * dx
        mid += fm * dx

    return (round(trap, 10), round(mid, 10), round(abs(trap - mid), 12))
