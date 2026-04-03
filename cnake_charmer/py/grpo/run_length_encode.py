"""Run-length encode a sequence of integers.

Keywords: grpo, compression, encoding, arrays, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def run_length_encode(n: int) -> tuple:
    """Run-length encode a deterministic integer sequence.

    Generates a repeating pattern and encodes consecutive runs.
    Returns (number of runs, total encoded length, checksum of run lengths).

    Args:
        n: Length of the input sequence.

    Returns:
        Tuple of (num_runs, total_length, checksum).
    """
    # Generate deterministic input with repetition patterns
    data = [0] * n
    for i in range(n):
        data[i] = (i * 7 + 13) % 20

    # Run-length encode
    if n == 0:
        return (0, 0, 0)

    num_runs = 1
    total_length = 0
    checksum = 0
    current_val = data[0]
    current_len = 1

    for i in range(1, n):
        if data[i] == current_val:
            current_len += 1
        else:
            total_length += current_len
            checksum = (checksum + current_len * (current_val + 1)) & 0xFFFFFFFF
            num_runs += 1
            current_val = data[i]
            current_len = 1

    # Final run
    total_length += current_len
    checksum = (checksum + current_len * (current_val + 1)) & 0xFFFFFFFF

    return (num_runs, total_length, checksum)
