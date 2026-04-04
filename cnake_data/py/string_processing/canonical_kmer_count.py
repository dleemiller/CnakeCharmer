"""
Count canonical DNA k-mers using reverse-complement canonicalization.

Sourced from SFT DuckDB blob: 883c81cb1a9b3dd302f4cd96835ffa76d931e2c8
Keywords: reverse complement, canonical kmer, dna, string processing
"""

from cnake_data.benchmarks import python_benchmark

_TRANS = str.maketrans("ACGTN", "TGCAN")


def _revcomp(seq: str) -> str:
    return seq.translate(_TRANS)[::-1]


def _canonical(seq: str) -> str:
    rc = _revcomp(seq)
    return seq if seq <= rc else rc


@python_benchmark(args=(240000, 11, 7))
def canonical_kmer_count(seq_len: int, k: int, stride: int) -> tuple:
    """Return (num_unique_canonical, total_kmers, gc_heavy_count)."""
    bases = "ACGT"
    text = "".join(bases[(i * stride + 3) % 4] for i in range(seq_len))
    freq = {}
    total = 0
    gc_heavy = 0

    for i in range(0, seq_len - k + 1):
        kmer = text[i : i + k]
        can = _canonical(kmer)
        freq[can] = freq.get(can, 0) + 1
        total += 1
        gc = can.count("G") + can.count("C")
        if gc * 2 >= k:
            gc_heavy += 1

    return (len(freq), total, gc_heavy)
