# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Jaro-Winkler string similarity distance.

Keywords: jaro winkler, string similarity, string distance, fuzzy matching, cython
"""

from libc.stdlib cimport malloc, free, calloc

from cnake_data.benchmarks import cython_benchmark


cdef double _jaro_similarity(const unsigned char *s1, int len1,
                              const unsigned char *s2, int len2) nogil:
    cdef int match_dist, matches, transpositions
    cdef int i, j, k, start, end
    cdef char *s1_matches
    cdef char *s2_matches

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = (len1 if len1 > len2 else len2) / 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = <char *>calloc(len1, sizeof(char))
    s2_matches = <char *>calloc(len2, sizeof(char))
    if not s1_matches or not s2_matches:
        free(s1_matches)
        free(s2_matches)
        return 0.0

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = i - match_dist
        if start < 0:
            start = 0
        end = i + match_dist + 1
        if end > len2:
            end = len2
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = 1
            s2_matches[j] = 1
            matches += 1
            break

    if matches == 0:
        free(s1_matches)
        free(s2_matches)
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    cdef double result = (<double>matches / len1 + <double>matches / len2 +
            (matches - transpositions / 2.0) / matches) / 3.0

    free(s1_matches)
    free(s2_matches)
    return result


cdef double _jaro_winkler(const unsigned char *s1, int len1,
                           const unsigned char *s2, int len2,
                           double scaling) nogil:
    cdef double jaro = _jaro_similarity(s1, len1, s2, len2)
    cdef int prefix = 0
    cdef int limit = len1
    if len2 < limit:
        limit = len2
    if limit > 4:
        limit = 4

    cdef int i
    for i in range(limit):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * scaling * (1.0 - jaro)


@cython_benchmark(syntax="cy", args=(150,))
def jaro_winkler(int n):
    """Compute Jaro-Winkler similarity for n*(n-1)/2 string pairs.

    Args:
        n: Number of strings to generate and compare.

    Returns:
        Tuple of (total_similarity, max_similarity, count_above_half).
    """
    cdef int i, j
    cdef int length
    cdef double sim
    cdef double total_sim = 0.0
    cdef double max_sim = 0.0
    cdef int count_above = 0

    # Generate n deterministic strings as bytes
    cdef int max_len = 21  # max possible: 8 + 12 = 20, +1 for safety
    cdef unsigned char *string_data = <unsigned char *>malloc(n * max_len * sizeof(unsigned char))
    cdef int *string_lens = <int *>malloc(n * sizeof(int))
    if not string_data or not string_lens:
        free(string_data)
        free(string_lens)
        raise MemoryError()

    for i in range(n):
        length = 8 + (i * 3) % 13
        string_lens[i] = length
        for j in range(length):
            string_data[i * max_len + j] = 97 + (i * 7 + j * 13 + 5) % 26

    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaro_winkler(
                &string_data[i * max_len], string_lens[i],
                &string_data[j * max_len], string_lens[j],
                0.1)
            total_sim += sim
            if sim > max_sim:
                max_sim = sim
            if sim > 0.5:
                count_above += 1

    free(string_data)
    free(string_lens)
    return (total_sim, max_sim, count_above)
