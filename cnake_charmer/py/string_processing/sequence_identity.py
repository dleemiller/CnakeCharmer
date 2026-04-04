"""Pairwise sequence identity computation over deterministic aligned sequence pairs.

Keywords: string processing, sequence identity, alignment, bioinformatics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


@python_benchmark(args=(10000,))
def sequence_identity(n: int) -> tuple:
    """Compute pairwise alignment identity for n sequence pairs.

    For each pair i, generates query and target strings of length 100 using
    a deterministic alphabet. Counts exact matches (excluding gap '-') and
    pairs with identity > 0.5.

    Args:
        n: Number of sequence pairs to evaluate.

    Returns:
        Tuple of (total_matches, pairs_above_half) as integers.
    """
    seq_len = 100
    total_matches = 0
    pairs_above_half = 0

    for i in range(n):
        matches = 0
        for j in range(seq_len):
            qc = _ALPHABET[(i * 7 + j * 13) % 21]
            tc = _ALPHABET[(i * 11 + j * 17) % 21]
            if qc == tc and qc != "-":
                matches += 1
        total_matches += matches
        if matches > 50:  # identity > 0.5 (matches / 100 > 0.5)
            pairs_above_half += 1

    return (total_matches, pairs_above_half)
