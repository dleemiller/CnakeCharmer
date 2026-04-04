# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generate pseudo-random nucleotide sequence counts from an LCG (Cython)."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(42, 3877, 29573, 139968, 600000, 0.60))
def fasta_lcg_gc_counts(
    unsigned long long seed,
    unsigned long long ia,
    unsigned long long ic,
    unsigned long long im,
    int length,
    double gc_threshold,
):
    cdef int gc_count = 0
    cdef int at_count = 0
    cdef int runs_over_threshold = 0
    cdef int current_run = 0
    cdef int i
    cdef unsigned long long x = seed
    cdef double r

    for i in range(length):
        x = (x * ia + ic) % im
        r = x / <double>im

        if r < 0.3029549426680:
            at_count += 1
            if current_run > 0 and (current_run / 11.0) > gc_threshold:
                runs_over_threshold += 1
            current_run = 0
        elif r < 0.5009432431601:
            gc_count += 1
            current_run += 1
        elif r < 0.6984905497992:
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
