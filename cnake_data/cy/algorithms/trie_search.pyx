# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Build a trie from deterministic words and count query matches (Cython-optimized).

Keywords: algorithms, trie, search, data structure, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def trie_search(int n):
    """Build a flat-array trie from n words and count query matches using C arrays.

    Words: word[i] = chars where ch[j] = (i*j+3)%26.
    Queries: query[i] = chars where ch[j] = (i*j+7)%26.

    Args:
        n: Number of words to insert and queries to search.

    Returns:
        Number of query words found in the trie.
    """
    cdef int ALPHABET = 26
    cdef int WORD_LEN = 5
    cdef int MAX_NODES = n * WORD_LEN + 1
    cdef int node_count = 1

    # Each node has 26 children (indices into pool) and a terminal flag
    cdef int *children = <int *>malloc(MAX_NODES * ALPHABET * sizeof(int))
    cdef char *terminal = <char *>malloc(MAX_NODES * sizeof(char))

    if children == NULL or terminal == NULL:
        if children != NULL:
            free(children)
        if terminal != NULL:
            free(terminal)
        raise MemoryError("Failed to allocate trie")

    memset(children, -1, MAX_NODES * ALPHABET * sizeof(int))
    memset(terminal, 0, MAX_NODES * sizeof(char))

    cdef int i, j, ch, cur, child_idx

    # Insert words: word[i] has char j = (i*j+3) % 26
    for i in range(n):
        cur = 0
        for j in range(WORD_LEN):
            ch = (i * j + 3) % 26
            child_idx = cur * ALPHABET + ch
            if children[child_idx] == -1:
                children[child_idx] = node_count
                node_count += 1
            cur = children[child_idx]
        terminal[cur] = 1

    # Count query matches: query[i] has char j = (i*j+7) % 26
    cdef int count = 0
    for i in range(n):
        cur = 0
        for j in range(WORD_LEN):
            ch = (i * j + 7) % 26
            child_idx = cur * ALPHABET + ch
            if children[child_idx] == -1:
                cur = -1
                break
            cur = children[child_idx]
        if cur != -1 and terminal[cur] == 1:
            count += 1

    free(children)
    free(terminal)
    return count
