import math


def hamiltonian_trig(n):
    """Build a tight-binding Hamiltonian matrix using cos/sin hopping terms.

    Returns (trace, max_element, sum_abs_elements).
    """
    ham = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if abs(i - j) == 1:
                ham[i][j] = math.cos(0.5 * (i + j)) + math.sin(0.3 * (i - j))
            elif i == j:
                ham[i][j] = 2.0 * math.cos(0.1 * i)

    trace = 0.0
    max_elem = 0.0
    sum_abs = 0.0
    for i in range(n):
        for j in range(n):
            trace += ham[i][j] if i == j else 0.0
            if abs(ham[i][j]) > max_elem:
                max_elem = abs(ham[i][j])
            sum_abs += abs(ham[i][j])

    return (round(trace, 6), round(max_elem, 6), round(sum_abs, 6))
