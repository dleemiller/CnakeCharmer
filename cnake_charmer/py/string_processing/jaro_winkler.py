"""Jaro-Winkler string similarity distance.

Keywords: jaro winkler, string similarity, string distance, fuzzy matching
"""

from cnake_charmer.benchmarks import python_benchmark


def _jaro_similarity(s1, s2):
    """Compute Jaro similarity between two strings."""
    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
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

    return (matches / len1 + matches / len2 + (matches - transpositions / 2.0) / matches) / 3.0


def _jaro_winkler(s1, s2, scaling=0.1):
    """Compute Jaro-Winkler similarity."""
    jaro = _jaro_similarity(s1, s2)

    # Common prefix length (up to 4)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * scaling * (1.0 - jaro)


@python_benchmark(args=(150,))
def jaro_winkler(n):
    """Compute Jaro-Winkler similarity for n*(n-1)/2 string pairs.

    Args:
        n: Number of strings to generate and compare.

    Returns:
        Tuple of (total_similarity, max_similarity, count_above_half).
    """
    # Generate n deterministic strings
    strings = []
    for i in range(n):
        length = 8 + (i * 3) % 13
        chars = []
        for j in range(length):
            chars.append(chr(97 + (i * 7 + j * 13 + 5) % 26))
        strings.append("".join(chars))

    total_sim = 0.0
    max_sim = 0.0
    count_above = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaro_winkler(strings[i], strings[j])
            total_sim += sim
            if sim > max_sim:
                max_sim = sim
            if sim > 0.5:
                count_above += 1

    return (total_sim, max_sim, count_above)
