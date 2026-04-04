"""
Canonical DNA k-mer counting via reverse complement comparison.

Keywords: string processing, dna, kmer, reverse complement, canonical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def dna_kmer_canonical(n: int) -> tuple:
    """Extract k-mers, canonicalize by comparing to reverse complement, count uniques.

    For each k-mer, compute its reverse complement (A<->T, C<->G) and pick
    the lexicographically smaller one. Track unique canonical k-mers and a
    checksum of first characters.

    Args:
        n: Length of the generated DNA sequence.

    Returns:
        Tuple of (num_unique_canonical, checksum) where checksum is sum of
        ord of first char of each canonical kmer across all positions.
    """
    bases = "ACGT"
    seq = "".join(bases[(i * 7 + 3) % 4] for i in range(n))

    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    k = 8
    seen = set()
    checksum = 0

    for i in range(n - k + 1):
        kmer = seq[i : i + k]
        # Build reverse complement char by char
        rc = ""
        for j in range(k - 1, -1, -1):
            rc += complement[kmer[j]]
        # Pick canonical (lexicographically smaller)
        canonical = kmer if kmer <= rc else rc
        seen.add(canonical)
        checksum += ord(canonical[0])

    return (len(seen), checksum)
