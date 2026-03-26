# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LZW compression dictionary size computation (Cython-optimized).

Keywords: compression, lzw, dictionary, encoding, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def lzw_compress(int n):
    """Count LZW dictionary entries using a trie with flat C arrays.

    Each trie node has 4 children (alphabet size = 4).
    Node pool allocated as flat array.
    """
    cdef int max_nodes = n + 4  # at most n new entries
    cdef int *children = <int *>malloc(max_nodes * 4 * sizeof(int))
    if not children:
        raise MemoryError()

    memset(children, -1, max_nodes * 4 * sizeof(int))

    cdef int next_node = 4  # first 4 nodes are single chars
    cdef int current, ch, i
    cdef int dict_size = 4

    # Generate first char
    current = (0 * 7 + 3) % 4

    for i in range(1, n):
        ch = (i * 7 + 3) % 4
        if children[current * 4 + ch] != -1:
            # Extend current match
            current = children[current * 4 + ch]
        else:
            # Add new entry
            children[current * 4 + ch] = next_node
            next_node += 1
            dict_size += 1
            current = ch  # Reset to single char

    free(children)
    return dict_size
