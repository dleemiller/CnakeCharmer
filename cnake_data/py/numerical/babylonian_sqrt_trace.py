"""
Babylonian square-root iteration with convergence trace.

Sourced from SFT DuckDB blob: 32dde1ea10e5554dbd70a4b13a8fb0b220436fd9
Keywords: babylonian sqrt, newton method, convergence, numerical
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(12345.6789, 1000000, 0.0))
def babylonian_sqrt_trace(value: float, loops: int, tolerance: float) -> tuple:
    """Approximate sqrt(value) and expose convergence diagnostics."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0.0:
        return (0.0, 0.0, 0)

    x = 1.0 if value < 1.0 else value * 0.5
    last_diff = 0.0
    used = 0

    for i in range(loops):
        x_prev = x
        x = 0.5 * (x + value / x)
        last_diff = abs(x - x_prev)
        used = i + 1
        if last_diff < tolerance:
            break

    return (round(x, 12), round(last_diff, 16), used)
