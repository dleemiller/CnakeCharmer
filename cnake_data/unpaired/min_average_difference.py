def _abs_diff(a, b):
    return a - b if a > b else b - a


def min_average_difference(buff_prev, buff_next):
    """For each frame in buff_next, find closest frame in buff_prev by L1 distance."""
    n_prev = len(buff_prev)
    n_curr = len(buff_next)
    n_row = len(buff_prev[0]) if n_prev else 0
    n_col = len(buff_prev[0][0]) if n_prev and n_row else 0

    max_int = 2**63 - 1
    min_avg = [0 for _ in range(n_curr)]
    min_avg_index = [0 for _ in range(n_curr)]

    for kn in range(n_curr):
        current_min = max_int
        current_ind = -1

        for kp in range(n_prev):
            total = 0
            for i in range(n_row):
                for j in range(n_col):
                    if total > current_min:
                        break
                    total += _abs_diff(buff_prev[kp][i][j], buff_next[kn][i][j])

            if total < current_min:
                current_min = total
                current_ind = kp

        min_avg[kn] = current_min
        min_avg_index[kn] = current_ind

    return min_avg, min_avg_index


def average_difference_matrix(buff_prev, buff_next):
    """Return full pairwise L1 distance matrix between frame buffers."""
    n_prev = len(buff_prev)
    n_curr = len(buff_next)
    n_row = len(buff_prev[0]) if n_prev else 0
    n_col = len(buff_prev[0][0]) if n_prev and n_row else 0

    out = [[0 for _ in range(n_curr)] for _ in range(n_prev)]

    for kn in range(n_curr):
        for kp in range(n_prev):
            total = 0
            for i in range(n_row):
                for j in range(n_col):
                    total += _abs_diff(buff_prev[kp][i][j], buff_next[kn][i][j])
            out[kp][kn] = total

    return out
