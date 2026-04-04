"""Sum of squares via buffer-backed double array.

Fills a list of doubles derived from index-based hashing, then
computes the sum of squares. Demonstrates the pure-Python baseline
for a Cython buffer-protocol cdef class.

Keywords: numerical, buffer protocol, sum of squares, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def buffer_sum_squares(n: int) -> float:
    """Compute sum of squares of hash-derived doubles.

    Args:
        n: Number of elements.

    Returns:
        Sum of squares as a float.
    """
    mask = 0xFFFFFFFF
    data = [0.0] * n
    for i in range(n):
        h = ((i * 2654435761) & mask) ^ ((i * 2246822519) & mask)
        data[i] = (h & 0xFFFF) / 65535.0

    total = 0.0
    for i in range(n):
        total += data[i] * data[i]
    return total
