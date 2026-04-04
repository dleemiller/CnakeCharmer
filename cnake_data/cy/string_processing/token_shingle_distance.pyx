# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Token shingle binary cosine and jaccard distances (Cython)."""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


@cython_benchmark(syntax="cy", args=(240000, 3, 26, 17))
def token_shingle_distance(int length, int ngram, int vocab, int seed):
    cdef int size = 32768
    cdef int mask = size - 1
    cdef int *a = <int *>malloc(length * sizeof(int))
    cdef int *b = <int *>malloc(length * sizeof(int))
    cdef int *ca = <int *>malloc(size * sizeof(int))
    cdef int *cb = <int *>malloc(size * sizeof(int))
    cdef unsigned int state = <unsigned int>((seed * 1664525 + 1013904223) & MASK32)
    cdef unsigned int h1 = 2166136261
    cdef unsigned int h2 = 2166136261
    cdef unsigned int mul = 16777619
    cdef int i, xa, xb, inter=0, union=0, na=0, nb=0
    cdef long long dot=0, sa=0, sb=0
    cdef double binary_cos, jaccard, weighted_cos

    if a == NULL or b == NULL or ca == NULL or cb == NULL:
        free(a); free(b); free(ca); free(cb)
        raise MemoryError()

    for i in range(size):
        ca[i] = 0
        cb[i] = 0

    for i in range(length):
        state = (state * 1664525 + 1013904223) & MASK32
        a[i] = state % vocab
        if (state & 31) == 0:
            b[i] = (a[i] + 1) % vocab
        else:
            b[i] = a[i]

    for i in range(ngram):
        h1 = ((h1 ^ <unsigned int>(a[i] + 1)) * mul) & MASK32
        h2 = ((h2 ^ <unsigned int>(b[i] + 1)) * mul) & MASK32
    ca[h1 & mask] += 1
    cb[h2 & mask] += 1

    for i in range(ngram, length):
        h1 = ((h1 ^ <unsigned int>(a[i] + 1) ^ <unsigned int>((a[i - ngram] + 1) << 1)) * mul) & MASK32
        h2 = ((h2 ^ <unsigned int>(b[i] + 1) ^ <unsigned int>((b[i - ngram] + 1) << 1)) * mul) & MASK32
        ca[h1 & mask] += 1
        cb[h2 & mask] += 1

    for i in range(size):
        xa = ca[i]
        xb = cb[i]
        if xa > 0: na += 1
        if xb > 0: nb += 1
        if xa > 0 and xb > 0: inter += 1
        if xa > 0 or xb > 0: union += 1
        dot += <long long>xa * xb
        sa += <long long>xa * xa
        sb += <long long>xb * xb

    binary_cos = inter / sqrt(na * nb) if na > 0 and nb > 0 else 0.0
    jaccard = inter / <double>union if union > 0 else 0.0
    weighted_cos = dot / sqrt(sa * sb) if sa > 0 and sb > 0 else 0.0

    free(a); free(b); free(ca); free(cb)
    return (binary_cos, jaccard, weighted_cos)
