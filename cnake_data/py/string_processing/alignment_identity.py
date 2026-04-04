"""
Sequence alignment identity computation for pairs of strings.

Computes the fraction of matching characters between aligned sequence pairs,
simulating a common bioinformatics operation on deterministic pseudo-random strings.

Keywords: string processing, alignment, identity, bioinformatics, sequence, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(20000,))
def alignment_identity(n: int) -> tuple:
    """Compute alignment identity for n pairs of 128-char sequences.

    Each sequence pair is deterministically generated from a 4-letter alphabet.
    Returns (sum of identities, count of pairs with identity > 0.5).

    Args:
        n: Number of sequence pairs to compare.

    Returns:
        Tuple of (sum of identity fractions, count of high-identity pairs).
    """
    alphabet = "ACGT"
    seq_len = 128
    total_identity = 0.0
    high_count = 0

    for i in range(n):
        matches = 0
        for j in range(seq_len):
            c1 = alphabet[((i * 31 + j * 7 + 3) * 2903) % 4]
            c2 = alphabet[((i * 37 + j * 11 + 5) * 3079) % 4]
            if c1 == c2:
                matches += 1
        identity = matches / seq_len
        total_identity += identity
        if identity > 0.5:
            high_count += 1

    return (total_identity, high_count)
