"""
Find poly-A tail positions in deterministic DNA sequences.

Scans each sequence from the 3' end to locate the start of poly-A/poly-N
tails and aggregates statistics.

Keywords: string processing, DNA, poly-A, tail finder, bioinformatics, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def poly_a_tail_finder(n: int) -> tuple:
    """Generate n deterministic DNA sequences and find poly-A tail positions.

    Each sequence is 64 characters long, built from a deterministic rule:
    position j gets base ACGTN based on (i*j + seed) mod values, with the
    last portion biased toward A/N to create poly-A tails.

    Returns (total_tail_lengths, count_with_tail) where a tail is defined
    as a contiguous run of A or N characters from the 3' end.

    Args:
        n: Number of DNA sequences to process.

    Returns:
        Tuple of (total_tail_lengths, count_with_tail).
    """
    bases = "ACGTN"
    seq_len = 64
    total_tail_lengths = 0
    count_with_tail = 0

    for i in range(n):
        # Build sequence deterministically
        # Last 16 chars biased toward A/N to create realistic tails
        seed = (i * 7 + 13) % 1000003

        # Scan from end to find poly-A tail (contiguous A or N from 3' end)
        tail_len = 0
        for j in range(seq_len - 1, -1, -1):
            if j >= seq_len - 16:
                # Biased region: higher chance of A/N
                val = (seed * (j + 1) + i * 3) % 7
                if val < 3:
                    base_idx = 0  # A
                elif val == 3:
                    base_idx = 4  # N
                else:
                    base_idx = (seed * j + i) % 4  # A, C, G, T
            else:
                # Normal region
                base_idx = (seed * (j + 1) + i * 11) % 5

            ch = bases[base_idx]
            if ch == "A" or ch == "N":
                tail_len = tail_len + 1
            else:
                break

        total_tail_lengths = total_tail_lengths + tail_len
        if tail_len > 0:
            count_with_tail = count_with_tail + 1

    return (total_tail_lengths, count_with_tail)
