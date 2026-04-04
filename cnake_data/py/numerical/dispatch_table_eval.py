"""Dispatch n evaluations through a table of math functions.

Keywords: numerical, dispatch, function pointer, function table, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _square(x):
    return x * x


def _cube(x):
    return x * x * x


def _negate(x):
    return -x


def _half(x):
    return x * 0.5


@python_benchmark(args=(100000,))
def dispatch_table_eval(n: int) -> float:
    """Dispatch n evaluations through a function table.

    Functions: [square, cube, negate, half].
    Input x = (i * 37 + 11) % 997 / 100.0.
    Function index = i % 4.
    Return sum of all results.

    Args:
        n: Number of evaluations.

    Returns:
        Sum of all dispatched function results.
    """
    funcs = [_square, _cube, _negate, _half]
    total = 0.0
    for i in range(n):
        x = ((i * 37 + 11) % 997) / 100.0
        func_idx = i % 4
        total += funcs[func_idx](x)

    return total
