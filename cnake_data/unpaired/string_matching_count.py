def string_matching_count(text_len, pattern_len):
    """Count character matches between a deterministic text and pattern using brute force.

    Generates a text and pattern from a repeating alphabet, then counts
    how many positions have at least one character match in a sliding window.

    Returns (match_count, total_comparisons).
    """
    alpha = "abcdefghijklmnop"
    text = ""
    for i in range(text_len):
        text += alpha[i % 16]
    pattern = ""
    for i in range(pattern_len):
        pattern += alpha[(i * 3 + 7) % 16]

    match_count = 0
    comparisons = 0

    for i in range(text_len - pattern_len + 1):
        found = True
        for j in range(pattern_len):
            comparisons += 1
            if text[i + j] != pattern[j]:
                found = False
                break
        if found:
            match_count += 1

    return (match_count, comparisons)
