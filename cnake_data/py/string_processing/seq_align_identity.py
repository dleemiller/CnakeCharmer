"""Sequence alignment identity computation.

Compares aligned DNA-like sequence pairs character by character,
counting matches excluding gaps to compute sequence identity.

Keywords: sequence alignment, identity, DNA, bioinformatics, string processing
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000, 15))
def seq_align_identity(n, gap_rate):
    """Compute alignment identity statistics over many sequence pairs.

    Generates deterministic aligned sequence pairs (DNA-like: ACGT with gaps '-')
    and computes match/mismatch/gap statistics for each pair.

    Args:
        n: Sequence length per pair.
        gap_rate: Every gap_rate-th position is a gap in one sequence.

    Returns:
        Tuple of (total_matches, total_identity_pct_sum, total_gaps).
    """
    bases = "ACGT"
    total_matches = 0
    total_identity_sum = 0.0
    total_gaps = 0
    n_pairs = 50

    for p in range(n_pairs):
        # Build two aligned sequences deterministically
        seq1 = []
        seq2 = []
        for i in range(n):
            seed = p * 997 + i * 31
            b1 = bases[seed % 4]
            b2 = bases[(seed * 7 + 3) % 4]
            # Insert gaps at regular intervals
            if i % gap_rate == (p % gap_rate):
                seq1.append("-")
                seq2.append(b2)
            elif i % gap_rate == ((p + 1) % gap_rate):
                seq1.append(b1)
                seq2.append("-")
            else:
                seq1.append(b1)
                seq2.append(b2)

        # Compute identity stats
        matches = 0
        gaps = 0
        aligned_len = 0
        for i in range(n):
            c1 = seq1[i]
            c2 = seq2[i]
            if c1 == "-" or c2 == "-":
                gaps += 1
            else:
                aligned_len += 1
                if c1 == c2:
                    matches += 1

        identity_pct = matches / aligned_len if aligned_len > 0 else 0.0
        total_matches += matches
        total_identity_sum += identity_pct
        total_gaps += gaps

    return (total_matches, total_identity_sum, total_gaps)
