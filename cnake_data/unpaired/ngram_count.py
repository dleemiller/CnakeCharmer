def ngram_count(n, gram_size=3):
    """Count distinct n-grams in a deterministic string of length n.

    Returns (num_distinct, total_ngrams, most_common_count).
    """
    alpha = "abcdefghijklmnop"
    text = ""
    for i in range(n):
        text += alpha[(i * 7 + 3) % 16]

    counts = {}
    total = 0
    for i in range(len(text) - gram_size + 1):
        gram = text[i : i + gram_size]
        counts[gram] = counts.get(gram, 0) + 1
        total += 1

    if not counts:
        return (0, 0, 0)

    most_common = max(counts.values())
    return (len(counts), total, most_common)
