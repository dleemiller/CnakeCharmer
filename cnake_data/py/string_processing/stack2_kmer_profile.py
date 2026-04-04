"""Profile fixed-width k-mer counts over generated DNA text.

Adapted from The Stack v2 Cython candidate:
- blob_id: 62e948730a990693030f1347641dec71b649050b
- filename: knucleotide.pyx

Keywords: string processing, dna, kmer, frequency, rolling hash
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(180000, 6, 31))
def stack2_kmer_profile(dna_length: int, motif_width: int, seed_tag: int) -> tuple:
    """Count k-mers using base-4 rolling hash and return summary statistics."""
    if motif_width <= 0 or dna_length < motif_width:
        return (0, 0, 0, 0)

    state = (2463534242 + seed_tag * 1223) & 0xFFFFFFFF
    dna_codes = [0] * dna_length
    for idx in range(dna_length):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        dna_codes[idx] = state & 3

    table_size = 1 << (2 * motif_width)
    counts = [0] * table_size
    mask = table_size - 1

    code = 0
    for idx in range(motif_width):
        code = (code << 2) | dna_codes[idx]
    counts[code] += 1

    for idx in range(motif_width, dna_length):
        code = ((code << 2) & mask) | dna_codes[idx]
        counts[code] += 1

    top_idx = 0
    top_count = counts[0]
    distinct = 0
    checksum = 0
    for idx, val in enumerate(counts):
        if val:
            distinct += 1
            checksum = (checksum + val * (idx + 7)) & 0xFFFFFFFF
            if val > top_count:
                top_count = val
                top_idx = idx

    return (distinct, top_idx, top_count, checksum)
