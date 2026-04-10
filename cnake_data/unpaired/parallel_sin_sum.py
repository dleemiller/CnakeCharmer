import math


def parallel_sin_sum(n):
    """Sum sin(i) and |sin(i)| for i in range(n).

    Returns (sin_sum, abs_sin_sum, count_positive).
    """
    sin_sum = 0.0
    abs_sum = 0.0
    count_pos = 0
    for i in range(n):
        val = math.sin(i)
        sin_sum += val
        abs_sum += abs(val)
        if val > 0:
            count_pos += 1
    return (round(sin_sum, 6), round(abs_sum, 6), count_pos)
