def factorial_combinations(n):
    """Compute n!, and count combinations C(n, k) for all k.

    Returns (factorial_n, sum_of_combinations, max_combination).
    """
    # Compute factorial
    fact = 1
    for i in range(1, n + 1):
        fact *= i

    # Compute all C(n, k) using Pascal's triangle row
    row = [1] * (n + 1)
    for i in range(1, n + 1):
        new_row = [1] * (n + 1)
        for j in range(1, i):
            new_row[j] = row[j - 1] + row[j]
        row = new_row

    sum_comb = sum(row[: n + 1])
    max_comb = max(row[: n + 1])
    return (fact, sum_comb, max_comb)
