"""
Copy an array, apply an in-place transform to the copy, and return the sum.

Keywords: numerical, copy, transform, array, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def memview_copy_transform(n: int) -> float:
    """Copy array, square each element in the copy, return sum.

    Data: data[i] = ((i * 29 + 5) % 200) / 10.0
    Transform: copy[i] = copy[i] * copy[i]

    Args:
        n: Length of the array.

    Returns:
        Sum of the transformed copy.
    """
    data = [0.0] * n
    for i in range(n):
        data[i] = ((i * 29 + 5) % 200) / 10.0

    # Copy
    copy = [0.0] * n
    for i in range(n):
        copy[i] = data[i]

    # Transform: square each element
    for i in range(n):
        copy[i] = copy[i] * copy[i]

    total = 0.0
    for i in range(n):
        total += copy[i]

    return total
