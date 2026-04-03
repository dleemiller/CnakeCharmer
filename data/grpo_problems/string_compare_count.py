def string_compare_count(n):
    """Count matching and mismatching characters between two deterministic strings.

    Returns (matches, mismatches, total_comparisons).
    """
    alpha = "abcdefghijklmnopqrstuvwxyz"
    a = ""
    b = ""
    for i in range(n):
        a += alpha[i % 26]
        b += alpha[(i * 3 + 5) % 26]

    matches = 0
    mismatches = 0
    length = min(len(a), len(b))
    for i in range(length):
        if a[i] == b[i]:
            matches += 1
        else:
            mismatches += 1
    return (matches, mismatches, length)
