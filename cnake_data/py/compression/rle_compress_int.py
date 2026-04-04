"""Run-length encode integers and count runs.

Keywords: run-length encoding, RLE, compression, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def rle_compress_int(n: int) -> int:
    """Run-length encode n integers and return the total number of runs.

    Values: v[i] = ((i * 37 + 7) // 4) % 10 (produces runs of varying length).

    Args:
        n: Number of integers.

    Returns:
        Number of runs as an integer.
    """
    if n == 0:
        return 0

    runs = 1
    prev = ((0 * 37 + 7) // 4) % 10

    for i in range(1, n):
        curr = ((i * 37 + 7) // 4) % 10
        if curr != prev:
            runs += 1
            prev = curr

    return runs
