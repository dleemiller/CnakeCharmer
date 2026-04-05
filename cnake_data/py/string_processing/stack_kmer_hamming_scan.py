"""Generate DNA kmers and compute bounded Hamming scan statistics.

Adapted from The Stack v2 Cython candidate:
- blob_id: 21f9a3240326d207eff6dda2a957d3f9116efabf
- filename: util.pyx

Keywords: string_processing, dna, kmer, hamming distance, scan
"""

from cnake_data.benchmarks import python_benchmark


def _early_hamming(a: str, b: str, max_d: int) -> int:
    d = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            d += 1
            if d > max_d:
                break
    return d


@python_benchmark(args=(40000, 11, 2))
def stack_kmer_hamming_scan(dna_len: int, k: int, max_d: int) -> tuple:
    """Build deterministic DNA sequence and summarize local kmer Hamming distances."""
    alpha = "ACGT"
    state = 123456789
    dna_chars = []
    for _ in range(dna_len):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        dna_chars.append(alpha[state & 3])
    dna = "".join(dna_chars)

    kmers = [dna[i : i + k] for i in range(dna_len - k + 1)]
    n = len(kmers)
    close = 0
    min_dist = k
    checksum = 0

    for i in range(1, n):
        d = _early_hamming(kmers[0], kmers[i], max_d)
        if d <= max_d:
            close += 1
        if d < min_dist:
            min_dist = d
        checksum = (checksum + d * (i + 17)) & 0xFFFFFFFF

    return (n, close, min_dist, checksum)
