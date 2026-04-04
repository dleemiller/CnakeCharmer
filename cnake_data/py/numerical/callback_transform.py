"""Apply different callback transformations to array elements.

Keywords: numerical, callback, transform, function pointer, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _scale_up(x):
    return x * 2.5


def _invert(x):
    return 1.0 / (1.0 + x * x)


def _smooth(x):
    return x / (1.0 + abs(x))


def _transform_array(arr, n, func):
    total = 0.0
    for i in range(n):
        total += func(arr[i])
    return total


@python_benchmark(args=(100000,))
def callback_transform(n: int) -> float:
    """Apply three different callbacks to an array, sum all results.

    Array: arr[i] = (i * 43 + 7) % 503 / 50.0.
    Apply scale_up, invert, and smooth to entire array. Sum all results.

    Args:
        n: Number of elements.

    Returns:
        Sum of all three transformation results.
    """
    arr = [0.0] * n
    for i in range(n):
        arr[i] = ((i * 43 + 7) % 503) / 50.0

    total = 0.0
    total += _transform_array(arr, n, _scale_up)
    total += _transform_array(arr, n, _invert)
    total += _transform_array(arr, n, _smooth)

    return total
