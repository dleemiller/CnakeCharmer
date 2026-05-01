from __future__ import annotations


def viterbi(
    emission_seq: list[list[float]], log_trans_t: list[list[float]]
) -> tuple[list[int], float]:
    t_len = len(emission_seq)
    n_states = len(emission_seq[0]) if t_len else 0
    ptr = [[0 for _ in range(n_states)] for _ in range(t_len)]
    prev = [emission_seq[0][s] + 1.0 + log_trans_t[s + 1][0] for s in range(n_states)]

    for t in range(t_len - 1):
        nxt = [0.0] * n_states
        for i in range(n_states):
            best_j = 0
            best_v = prev[0] + log_trans_t[i + 1][1]
            for j in range(1, n_states):
                cand = prev[j] + log_trans_t[i + 1][j + 1]
                if cand > best_v:
                    best_v = cand
                    best_j = j
            ptr[t + 1][i] = best_j
            nxt[i] = best_v + emission_seq[t + 1][i]
        prev = nxt

    end_vals = [prev[i] + log_trans_t[0][i + 1] for i in range(n_states)]
    last = max(range(n_states), key=lambda i: end_vals[i])
    llik = end_vals[last]
    path = [0] * t_len
    path[-1] = last
    for i in range(t_len - 1, 0, -1):
        path[i - 1] = ptr[i][path[i]]
    return path, llik
