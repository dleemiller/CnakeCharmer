# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
"""Locate poly-A/poly-N tail starts, poly-T/poly-N head ends, and motif hits (Cython).

Sourced from SFT DuckDB blob: a9af1de305dc057bf469a1b28266bfcb34c6e7d8
Keywords: dna, poly-a tail, poly-t head, motif, string processing, cython
"""

from libc.stdlib cimport free, malloc

from cnake_charmer.benchmarks import cython_benchmark


cdef inline int _find_poly_a_start_codes(const unsigned char *seq, int n):
    cdef int start
    # A=0, N=4
    for start in range(n, 0, -1):
        if seq[start - 1] != 0 and seq[start - 1] != 4:
            return start
    return 0


cdef inline int _find_poly_t_end_codes(const unsigned char *seq, int n):
    cdef int end
    # T=3, N=4
    for end in range(n):
        if seq[end] != 3 and seq[end] != 4:
            return end - 1
    return n - 1


cdef void _fill_seq(unsigned char *seq, int seq_len, int i, int motif_shift):
    cdef int j, idx
    for j in range(seq_len):
        if j < 8:
            idx = 3 if (i + j + motif_shift) % 4 < 3 else 4
        elif j >= seq_len - 10:
            idx = 0 if (i * 5 + j) % 4 < 3 else 4
        else:
            idx = (i * 7 + j * 11 + motif_shift) % 5
        seq[j] = <unsigned char>idx


cdef int _first_cagta(const unsigned char *seq, int seq_len):
    cdef int p
    for p in range(seq_len - 4):
        if (
            seq[p] == 1
            and seq[p + 1] == 0
            and seq[p + 2] == 2
            and seq[p + 3] == 3
            and seq[p + 4] == 0
        ):
            return p
    return -1


@cython_benchmark(syntax="cy", args=(60000, 72, 9))
def poly_tail_trim_indices(int seq_count, int seq_len, int motif_shift):
    cdef int i
    cdef long long sum_a = 0
    cdef long long sum_t = 0
    cdef int motif_hits = 0
    cdef int first_hit
    cdef unsigned char *seq = NULL

    if seq_len <= 0 or seq_count <= 0:
        return (0, 0, 0)

    seq = <unsigned char *>malloc(seq_len * sizeof(unsigned char))
    if seq == NULL:
        raise MemoryError()

    for i in range(seq_count):
        _fill_seq(seq, seq_len, i, motif_shift)

        sum_a += _find_poly_a_start_codes(seq, seq_len)
        sum_t += _find_poly_t_end_codes(seq, seq_len)

        first_hit = _first_cagta(seq, seq_len)
        if first_hit == 11 or first_hit == 12 or first_hit == 13:
            motif_hits += 1
    free(seq)

    return (sum_a, sum_t, motif_hits)
