"""
K-mer (dinucleotide) frequency counting on a synthetic DNA sequence.

Keywords: string processing, kmer, dna, frequency, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def kmer_frequency(n: int) -> tuple:
    """Count all 2-mer (dinucleotide) frequencies in a synthetic DNA string.

    Generate a DNA string of length n using an LCG:
        seq[i] = "ACGT"[(i*1664525 + 1013904223) & 3]

    Encode each 2-mer as idx = base1*4 + base2 (A=0, C=1, G=2, T=3).

    Args:
        n: Length of the DNA string.

    Returns:
        (total_2mer_count, most_frequent_2mer_idx, most_frequent_2mer_count)
    """
    counts = [0] * 16

    # Generate sequence and count 2-mers simultaneously
    # prev base index
    prev = (1013904223) & 3  # i=0: (0*1664525 + 1013904223) & 3

    for i in range(1, n):
        curr = (i * 1664525 + 1013904223) & 3
        counts[prev * 4 + curr] += 1
        prev = curr

    total = n - 1
    most_freq_idx = 0
    most_freq_count = counts[0]
    for k in range(1, 16):
        if counts[k] > most_freq_count:
            most_freq_count = counts[k]
            most_freq_idx = k

    return (total, most_freq_idx, most_freq_count)
