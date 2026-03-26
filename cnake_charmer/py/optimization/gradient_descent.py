"""Minimize Rosenbrock function using gradient descent.

Keywords: gradient descent, rosenbrock, optimization, minimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def gradient_descent(n: int) -> float:
    """Minimize Rosenbrock f(x,y) = 100*(y-x^2)^2 + (1-x)^2 via gradient descent.

    n iterations, lr=0.001. Start at (-1, 1). Returns final f value.

    Args:
        n: Number of gradient descent iterations.

    Returns:
        Final objective function value.
    """
    x = -1.0
    y = 1.0
    lr = 0.001

    for _ in range(n):
        # Gradient of Rosenbrock
        dx = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x)
        dy = 200.0 * (y - x * x)
        x -= lr * dx
        y -= lr * dy

    f_val = 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
    return f_val
