"""Deterministic simulated annealing on Rastrigin function with fixed schedule.

Keywords: optimization, simulated annealing, Rastrigin, deterministic, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def simulated_annealing_deterministic(n: int) -> tuple:
    """Deterministic simulated annealing to minimize 2D Rastrigin function.

    Uses a fixed deterministic perturbation schedule instead of random numbers.
    Perturbation at step i: dx = 0.1 * cos(i*0.1), dy = 0.1 * sin(i*0.13).
    Temperature schedule: T = 10.0 * (1 - i/n).
    Accept if f_new < f_old or if exp(-(f_new-f_old)/T) > threshold
    where threshold = 0.5 + 0.3*sin(i*0.07).

    Args:
        n: Number of annealing steps.

    Returns:
        Tuple of (final_x, final_y, best_f_value).
    """
    A = 10.0
    x = 2.5
    y = 2.5

    def rastrigin(xv, yv):
        return (
            2.0 * A
            + (xv * xv - A * math.cos(2.0 * math.pi * xv))
            + (yv * yv - A * math.cos(2.0 * math.pi * yv))
        )

    f_curr = rastrigin(x, y)
    best_x = x
    best_y = y
    best_f = f_curr

    for i in range(n):
        t_frac = i / float(n)
        temp = 10.0 * (1.0 - t_frac)

        dx = 0.1 * (1.0 - t_frac) * math.cos(i * 0.1)
        dy = 0.1 * (1.0 - t_frac) * math.sin(i * 0.13)

        x_new = x + dx
        y_new = y + dy

        f_new = rastrigin(x_new, y_new)
        delta = f_new - f_curr

        accept = False
        if delta < 0.0:
            accept = True
        elif temp > 1e-10:
            threshold = 0.5 + 0.3 * math.sin(i * 0.07)
            boltz = math.exp(-delta / temp) if delta / temp < 50.0 else 0.0
            if boltz > threshold:
                accept = True

        if accept:
            x = x_new
            y = y_new
            f_curr = f_new
            if f_curr < best_f:
                best_f = f_curr
                best_x = x
                best_y = y

    return (best_x, best_y, best_f)
