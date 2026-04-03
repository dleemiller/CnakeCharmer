"""K-mer frequency counting for DNA-like sequences.

Keywords: kmer, frequency, dna, sequence analysis, substring counting
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def kmer_frequency(n):
    """Count k-mer frequencies in a deterministic sequence of length n.

    Args:
        n: Sequence length.

    Returns:
        Tuple of (num_unique_3mers, most_frequent_count, total_pairs).
    """
    alphabet = "ACGT"

    # Generate deterministic DNA-like sequence
    seq = []
    for i in range(n):
        seq.append(alphabet[(i * 7 + 13) % 4])

    # Count 3-mers
    freq3 = {}
    for i in range(n - 2):
        kmer = seq[i] + seq[i + 1] + seq[i + 2]
        if kmer in freq3:
            freq3[kmer] += 1
        else:
            freq3[kmer] = 1

    num_unique = len(freq3)
    most_frequent = 0
    for count in freq3.values():
        if count > most_frequent:
            most_frequent = count

    # Count 5-mer pair co-occurrences within window of 20
    total_pairs = 0
    freq5 = {}
    for i in range(n - 4):
        kmer = seq[i] + seq[i + 1] + seq[i + 2] + seq[i + 3] + seq[i + 4]
        if kmer in freq5:
            freq5[kmer] += 1
        else:
            freq5[kmer] = 1

    for count in freq5.values():
        total_pairs += count * (count - 1) // 2

    return (num_unique, most_frequent, total_pairs)
