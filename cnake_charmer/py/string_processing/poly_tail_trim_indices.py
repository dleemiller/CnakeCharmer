"""
Locate poly-A/poly-N tail starts, poly-T/poly-N head ends, and motif hits.

Sourced from SFT DuckDB blob: a9af1de305dc057bf469a1b28266bfcb34c6e7d8
Keywords: dna, poly-a tail, poly-t head, motif, string processing
"""

from cnake_charmer.benchmarks import python_benchmark


def _find_poly_a_start(seq: str) -> int:
    for start in range(len(seq), 0, -1):
        if seq[start - 1] not in ("A", "N"):
            return start
    return 0


def _find_poly_t_end(seq: str) -> int:
    for end in range(len(seq)):
        if seq[end] not in ("T", "N"):
            return end - 1
    return len(seq) - 1


@python_benchmark(args=(60000, 72, 9))
def poly_tail_trim_indices(seq_count: int, seq_len: int, motif_shift: int) -> tuple:
    """Return (sum_poly_a_starts, sum_poly_t_ends, motif_hits)."""
    bases = "ACGTN"
    sum_a = 0
    sum_t = 0
    motif_hits = 0

    for i in range(seq_count):
        chars = []
        for j in range(seq_len):
            if j < 8:
                idx = 3 if (i + j + motif_shift) % 4 < 3 else 4
            elif j >= seq_len - 10:
                idx = 0 if (i * 5 + j) % 4 < 3 else 4
            else:
                idx = (i * 7 + j * 11 + motif_shift) % 5
            chars.append(bases[idx])
        seq = "".join(chars)

        sum_a += _find_poly_a_start(seq)
        sum_t += _find_poly_t_end(seq)
        if seq.find("CAGTA") in (11, 12, 13):
            motif_hits += 1

    return (sum_a, sum_t, motif_hits)
