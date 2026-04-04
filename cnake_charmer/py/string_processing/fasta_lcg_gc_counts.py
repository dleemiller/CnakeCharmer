"""Generate pseudo-random nucleotide sequence counts from an LCG.

Keywords: string processing, fasta, lcg, gc content, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MASK64 = (1 << 64) - 1


@python_benchmark(args=(42, 3877, 29573, 139968, 600000, 0.60))
def fasta_lcg_gc_counts(
    seed: int, ia: int, ic: int, im: int, length: int, gc_threshold: float
) -> tuple:
    """Generate nucleotide statistics from a deterministic LCG stream."""
    gc_count = 0
    at_count = 0
    runs_over_threshold = 0
    current_run = 0

    x = seed
    for _ in range(length):
        x = ((x * ia + ic) & MASK64) % im
        r = x / im

        if r < 0.3029549426680:
            gc = 0
        elif (
            r < 0.3029549426680 + 0.1979883004921
            or r < 0.3029549426680 + 0.1979883004921 + 0.1975473066391
        ):
            gc = 1
        else:
            gc = 0

        if gc:
            gc_count += 1
            current_run += 1
        else:
            at_count += 1
            if current_run > 0 and (current_run / 11.0) > gc_threshold:
                runs_over_threshold += 1
            current_run = 0

    if current_run > 0 and (current_run / 11.0) > gc_threshold:
        runs_over_threshold += 1

    return (gc_count, at_count, runs_over_threshold)
