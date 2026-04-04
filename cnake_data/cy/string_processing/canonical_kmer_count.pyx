# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count canonical DNA k-mers via reverse-complement canonicalization (Cython).

Sourced from SFT DuckDB blob: 883c81cb1a9b3dd302f4cd96835ffa76d931e2c8
Keywords: reverse complement, canonical kmer, dna, string processing, cython
"""

from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(240000, 11, 7))
def canonical_kmer_count(int seq_len, int k, int stride):
    cdef int i, j
    cdef int total = 0
    cdef int gc_heavy = 0
    cdef int gc, base_code
    cdef int unique = 0
    cdef unsigned char *seq = NULL
    cdef int *freq = NULL
    cdef int keyspace
    cdef int fwd, rev, can, t

    if seq_len <= 0 or k <= 0 or k > seq_len:
        return (0, 0, 0)
    if k > 15:
        raise ValueError("k must be <= 15")

    keyspace = 1 << (2 * k)  # 4**k
    seq = <unsigned char *>malloc(seq_len * sizeof(unsigned char))
    freq = <int *>malloc(keyspace * sizeof(int))
    if seq == NULL or freq == NULL:
        if seq != NULL:
            free(seq)
        if freq != NULL:
            free(freq)
        raise MemoryError()

    try:
        for i in range(keyspace):
            freq[i] = 0

        # Deterministic A/C/G/T stream encoded as 0..3.
        for i in range(seq_len):
            seq[i] = <unsigned char>((i * stride + 3) % 4)

        for i in range(seq_len - k + 1):
            fwd = 0
            rev = 0
            for j in range(k):
                fwd = (fwd << 2) | seq[i + j]
                base_code = seq[i + k - 1 - j]
                rev = (rev << 2) | (3 - base_code)  # reverse complement code

            can = fwd if fwd <= rev else rev
            if freq[can] == 0:
                unique += 1
            freq[can] += 1
            total += 1

            gc = 0
            t = can
            for j in range(k):
                base_code = t & 3
                if base_code == 1 or base_code == 2:
                    gc += 1
                t >>= 2
            if gc * 2 >= k:
                gc_heavy += 1
    finally:
        free(seq)
        free(freq)

    return (unique, total, gc_heavy)
