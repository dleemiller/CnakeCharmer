"""Gradient descent on Rosenbrock function from n starting points.

Runs gradient descent from multiple deterministic starting points and
returns the best result found.

Keywords: gradient descent, rosenbrock, optimization, minimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def gradient_descent(n: int) -> tuple:
    """Minimize Rosenbrock f(x,y) = 100*(y-x^2)^2 + (1-x)^2 via gradient descent.

    Runs 500 iterations from each of n starting points generated deterministically.

    Args:
        n: Number of starting points to try.

    Returns:
        Tuple of (best_x, best_y, best_val).
    """
    best_x = 0.0
    best_y = 0.0
    best_val = 1e30
    lr = 0.001
    iters = 500

    for s in range(n):
        # Deterministic starting point
        x = -2.0 + 4.0 * ((s * 7 + 3) % n) / n
        y = -2.0 + 4.0 * ((s * 13 + 7) % n) / n

        for _ in range(iters):
            # Gradient of Rosenbrock
            dx = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x)
            dy = 200.0 * (y - x * x)
            x -= lr * dx
            y -= lr * dy

        f_val = 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
        if f_val < best_val:
            best_val = f_val
            best_x = x
            best_y = y

    return (best_x, best_y, best_val)
