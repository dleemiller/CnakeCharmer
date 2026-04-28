import math


def tight_loop(
    array_len, input_2d_array, input_1d_array, max_iterations=100, d=0.9, threshold=1e-4
):
    """Iterative score propagation loop with convergence check."""
    score = [1.0] * array_len
    pscore = [1.0] * array_len

    for _ in range(max_iterations):
        pscore[:] = score[:]

        for i in range(array_len):
            summation = 0.0
            for j in range(array_len):
                wij = input_2d_array[i][j]
                if wij != 0:
                    summation += (wij / input_1d_array[j]) * score[j]
            score[i] = (1.0 - d) + d * summation

        diff_sum = 0.0
        for k in range(array_len):
            diff_sum += math.fabs(pscore[k] - score[k])

        if diff_sum <= threshold:
            break

    return score
