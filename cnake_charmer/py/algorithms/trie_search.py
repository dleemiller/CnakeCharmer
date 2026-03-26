"""
Build a trie from deterministic words and count how many queries are found.

Keywords: algorithms, trie, search, data structure, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def trie_search(n: int) -> int:
    """Build a trie from n deterministic words, then count how many queries match.

    Words are generated as word[i] = "".join(chr(65 + (i*j+3)%26) for j in range(5)).
    Queries are query[i] = "".join(chr(65 + (i*j+7)%26) for j in range(5)).

    Args:
        n: Number of words to insert and number of queries to search.

    Returns:
        Number of query words found in the trie.
    """
    # Build trie as nested dicts
    root = {}
    for i in range(n):
        node = root
        for j in range(5):
            ch = (i * j + 3) % 26
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node[-1] = True  # terminal marker

    # Count query matches
    count = 0
    for i in range(n):
        node = root
        found = True
        for j in range(5):
            ch = (i * j + 7) % 26
            if ch not in node:
                found = False
                break
            node = node[ch]
        if found and -1 in node:
            count += 1

    return count
