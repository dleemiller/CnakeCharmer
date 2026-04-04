"""Run-length encoding of a deterministic integer array.

Keywords: algorithms, run length encoding, compression, counting, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def run_length_encoding(n: int) -> int:
    """Count the number of runs in RLE of arr[i] = ((i*3+7)//5) % 10.

    Args:
        n: Size of the array.

    Returns:
        Total number of runs.
    """
    if n == 0:
        return 0

    runs = 1
    prev = ((0 * 3 + 7) // 5) % 10
    for i in range(1, n):
        val = ((i * 3 + 7) // 5) % 10
        if val != prev:
            runs += 1
            prev = val
    return runs
