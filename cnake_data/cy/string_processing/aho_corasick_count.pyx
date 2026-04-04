# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count pattern matches using Aho-Corasick automaton (Cython-optimized).

Keywords: aho-corasick, string matching, automaton, pattern, multi-pattern, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark

# Max nodes in automaton: sum of pattern lengths + 1
DEF MAX_NODES = 64
# Alphabet: A=0, B=1, C=2, D=3
DEF ALPHA_SIZE = 4


@cython_benchmark(syntax="cy", args=(1000000,))
def aho_corasick_count(int n):
    """Count total matches of 10 fixed patterns in a deterministic text."""
    # Automaton arrays: goto_table[node][char], fail[node], output_count[node]
    cdef int goto_table[MAX_NODES][ALPHA_SIZE]
    cdef int fail_arr[MAX_NODES]
    cdef int output_arr[MAX_NODES]
    cdef int queue[MAX_NODES]
    cdef int node_count, cur, ch_idx, qi, qe, r, s, state
    cdef long long total
    cdef int i

    # Initialize
    memset(goto_table, -1, MAX_NODES * ALPHA_SIZE * sizeof(int))
    memset(fail_arr, 0, MAX_NODES * sizeof(int))
    memset(output_arr, 0, MAX_NODES * sizeof(int))
    node_count = 1

    # Patterns as sequences of char indices (A=0, B=1, C=2, D=3)
    # AB, BA, AA, BB, ABC, BAC, ABA, BAB, ABAB, BABA
    cdef int patterns[10][4]
    cdef int pat_lens[10]

    patterns[0][0] = 0; patterns[0][1] = 1;  pat_lens[0] = 2  # AB
    patterns[1][0] = 1; patterns[1][1] = 0;  pat_lens[1] = 2  # BA
    patterns[2][0] = 0; patterns[2][1] = 0;  pat_lens[2] = 2  # AA
    patterns[3][0] = 1; patterns[3][1] = 1;  pat_lens[3] = 2  # BB
    patterns[4][0] = 0; patterns[4][1] = 1; patterns[4][2] = 2; pat_lens[4] = 3  # ABC
    patterns[5][0] = 1; patterns[5][1] = 0; patterns[5][2] = 2; pat_lens[5] = 3  # BAC
    patterns[6][0] = 0; patterns[6][1] = 1; patterns[6][2] = 0; pat_lens[6] = 3  # ABA
    patterns[7][0] = 1; patterns[7][1] = 0; patterns[7][2] = 1; pat_lens[7] = 3  # BAB
    patterns[8][0] = 0; patterns[8][1] = 1; patterns[8][2] = 0; patterns[8][3] = 1; pat_lens[8] = 4  # ABAB
    patterns[9][0] = 1; patterns[9][1] = 0; patterns[9][2] = 1; patterns[9][3] = 0; pat_lens[9] = 4  # BABA

    # Build trie
    cdef int p, k
    for p in range(10):
        cur = 0
        for k in range(pat_lens[p]):
            ch_idx = patterns[p][k]
            if goto_table[cur][ch_idx] == -1:
                goto_table[cur][ch_idx] = node_count
                node_count += 1
            cur = goto_table[cur][ch_idx]
        output_arr[cur] += 1

    # Set root transitions for missing chars to 0
    for ch_idx in range(ALPHA_SIZE):
        if goto_table[0][ch_idx] == -1:
            goto_table[0][ch_idx] = 0

    # Build fail links via BFS
    qi = 0
    qe = 0
    for ch_idx in range(ALPHA_SIZE):
        s = goto_table[0][ch_idx]
        if s != 0:
            fail_arr[s] = 0
            queue[qe] = s
            qe += 1

    while qi < qe:
        r = queue[qi]
        qi += 1
        for ch_idx in range(ALPHA_SIZE):
            s = goto_table[r][ch_idx]
            if s == -1:
                # Follow fail links
                goto_table[r][ch_idx] = goto_table[fail_arr[r]][ch_idx]
            else:
                fail_arr[s] = goto_table[fail_arr[r]][ch_idx]
                if fail_arr[s] == s:
                    fail_arr[s] = 0
                output_arr[s] += output_arr[fail_arr[s]]
                queue[qe] = s
                qe += 1

    # Allocate text array
    cdef char *text = <char *>malloc(n * sizeof(char))
    if not text:
        raise MemoryError()

    for i in range(n):
        text[i] = (i * 7 + 3) % 4

    # Search
    total = 0
    state = 0
    for i in range(n):
        state = goto_table[state][<int>text[i]]
        total += output_arr[state]

    free(text)

    return int(total)
