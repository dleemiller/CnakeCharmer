def triangular_number(n):
    """Compute the nth triangular number by iterative summation.

    Also computes running statistics. Returns (triangular, sum_of_squares, count_even).
    """
    total = 0
    sum_sq = 0
    count_even = 0
    for i in range(1, n + 1):
        total += i
        sum_sq += i * i
        if i % 2 == 0:
            count_even += 1
    return (total, sum_sq, count_even)
