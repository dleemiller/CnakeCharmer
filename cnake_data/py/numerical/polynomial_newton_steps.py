"""
Newton-Raphson iteration on a cubic polynomial with trace statistics.

Sourced from SFT DuckDB blob: 055f6686df5aedcd5115738a7d744e3de7d821e7
Keywords: newton raphson, polynomial root, numerical method, iteration
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1.75, 2000000, 1.0))
def polynomial_newton_steps(start: float, iterations: int, cubic_scale: float) -> tuple:
    """Run Newton steps on f(x)=cubic_scale*x^3-2x-1 and return trace metrics."""
    x = start
    step_sum = 0.0

    for _ in range(iterations):
        fx = cubic_scale * x * x * x - 2.0 * x - 1.0
        dfx = 3.0 * cubic_scale * x * x - 2.0
        if dfx == 0.0:
            break
        x_next = x - fx / dfx
        step_sum += abs(x_next - x)
        x = x_next

    fx_final = cubic_scale * x * x * x - 2.0 * x - 1.0
    avg_step = step_sum / iterations if iterations > 0 else 0.0
    return (round(x, 12), round(fx_final, 12), round(avg_step, 12))
