"""Count pattern matches using Aho-Corasick automaton.

Keywords: aho-corasick, string matching, automaton, pattern, multi-pattern, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def aho_corasick_count(n: int) -> int:
    """Count total matches of 10 fixed patterns in a deterministic text.

    Text: chr(65 + (i*7+3) % 4) for length n (chars A-D).
    Patterns: AB, BA, AA, BB, ABC, BAC, ABA, BAB, ABAB, BABA.

    Args:
        n: Length of the text.

    Returns:
        Total number of pattern matches found.
    """
    patterns = ["AB", "BA", "AA", "BB", "ABC", "BAC", "ABA", "BAB", "ABAB", "BABA"]
    text = [chr(65 + (i * 7 + 3) % 4) for i in range(n)]

    # Build Aho-Corasick automaton
    # Each node: dict of children, fail link, output count
    goto = [{}]
    fail = [0]
    output = [0]
    node_count = 1

    # Build trie
    for pattern in patterns:
        cur = 0
        for ch in pattern:
            if ch not in goto[cur]:
                goto.append({})
                fail.append(0)
                output.append(0)
                goto[cur][ch] = node_count
                node_count += 1
            cur = goto[cur][ch]
        output[cur] += 1

    # Build fail links via BFS
    queue = []
    for _ch, next_node in goto[0].items():
        fail[next_node] = 0
        queue.append(next_node)

    qi = 0
    while qi < len(queue):
        r = queue[qi]
        qi += 1
        for ch, s in goto[r].items():
            queue.append(s)
            state = fail[r]
            while state != 0 and ch not in goto[state]:
                state = fail[state]
            fail[s] = goto[state].get(ch, 0)
            if fail[s] == s:
                fail[s] = 0
            output[s] += output[fail[s]]

    # Search
    total = 0
    state = 0
    for ch in text:
        while state != 0 and ch not in goto[state]:
            state = fail[state]
        state = goto[state].get(ch, 0)
        total += output[state]

    return total
