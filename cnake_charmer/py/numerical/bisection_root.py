"""Generic bisection root finder for multiple functions.

Keywords: numerical, bisection, root finding, callback, function pointer, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _f1(x):
    return x * x - 2.0


def _f2(x):
    return x * x * x - x - 1.0


def _f3(x):
    return x * x - 4.0 * x + 3.5


def _f4(x):
    return x * x * x - 6.0 * x * x + 11.0 * x - 5.5


def _bisect(func, a, b, max_iter):
    """Bisection method to find root of func in [a, b]."""
    for _ in range(max_iter):
        mid = (a + b) * 0.5
        if func(mid) * func(a) <= 0.0:
            b = mid
        else:
            a = mid
    return (a + b) * 0.5


@python_benchmark(args=(10000,))
def bisection_root(n: int) -> float:
    """Find roots of 4 functions using bisection, repeated n times.

    Functions: x^2-2, x^3-x-1, x^2-4x+3.5, x^3-6x^2+11x-5.5.
    Each bisection runs 50 iterations per call.
    Returns sum of all found roots * n.

    Args:
        n: Number of repetitions.

    Returns:
        Sum of all root approximations.
    """
    funcs = [_f1, _f2, _f3, _f4]
    intervals = [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (0.5, 1.5)]

    total = 0.0
    for _ in range(n):
        for k in range(4):
            root = _bisect(funcs[k], intervals[k][0], intervals[k][1], 50)
            total += root

    return total
