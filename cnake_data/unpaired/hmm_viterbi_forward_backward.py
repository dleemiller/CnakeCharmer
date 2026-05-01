from __future__ import annotations


def _max_argmax_row(arr: list[float]) -> tuple[int, float]:
    pos = 0
    m = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > m:
            m = arr[i]
            pos = i
    return pos, m


def viterbi_log(
    emission_seq: list[list[float]], log_trans_t: list[list[float]]
) -> tuple[list[int], float]:
    t_len = len(emission_seq)
    n_states = len(emission_seq[0])
    ptr = [[0 for _ in range(n_states)] for _ in range(t_len)]
    prev = [emission_seq[0][s] + 1.0 + log_trans_t[s + 1][0] for s in range(n_states)]

    for t in range(1, t_len):
        new_prev = [0.0] * n_states
        for s in range(n_states):
            vals = [prev[j] + log_trans_t[s + 1][j + 1] for j in range(n_states)]
            arg, mx = _max_argmax_row(vals)
            ptr[t][s] = arg
            new_prev[s] = mx + emission_seq[t][s]
        prev = new_prev

    end_trans = [log_trans_t[0][i + 1] for i in range(n_states)]
    final_vals = [prev[i] + end_trans[i] for i in range(n_states)]
    last_state, llik = _max_argmax_row(final_vals)

    path = [0] * t_len
    path[-1] = last_state
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = ptr[t][path[t]]
    return path, llik
