# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Embedding lookup on f32 tensor (basic Cython, scalar loop).

Keywords: embedding, lookup, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def embedding_lookup(int n):
    """Lookup n embeddings from table, return sum of all values."""
    cdef int vocab_size = 1000
    cdef int dim = 64
    cdef float *table = <float *>malloc(vocab_size * dim * sizeof(float))
    if not table:
        raise MemoryError()

    cdef int v, d, i, idx
    cdef double total = 0.0
    cdef float *row

    # Build embedding table
    for v in range(vocab_size):
        for d in range(dim):
            table[v * dim + d] = sin((v * dim + d) * 0.01) * 0.1

    # Lookup and sum
    for i in range(n):
        idx = (i * 7 + 3) % vocab_size
        row = &table[idx * dim]
        for d in range(dim):
            total += row[d]

    free(table)
    return total
