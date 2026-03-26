"""
Run-length encode a deterministic string and count the number of runs.

Keywords: string processing, run-length encoding, compression, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def run_length_encode(n: int) -> int:
    """Count the number of runs in a run-length encoding of a deterministic string.

    The string is generated as s = "".join(chr(65 + (i*3)%5) for i in range(n)).
    A run is a maximal sequence of identical consecutive characters.

    Args:
        n: Length of the string to generate.

    Returns:
        Total number of runs as an integer.
    """
    s = "".join(chr(65 + (i * 3) % 5) for i in range(n))

    if n == 0:
        return 0

    runs = 1
    for i in range(1, n):
        if s[i] != s[i - 1]:
            runs += 1

    return runs
