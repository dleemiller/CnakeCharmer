from __future__ import annotations


def get_smolyak_coefficients(subspace_indices: list[list[int]]) -> list[float]:
    num_vars = len(subspace_indices)
    num_subspaces = len(subspace_indices[0]) if num_vars else 0
    out = [0.0] * num_subspaces
    for ii in range(num_subspaces):
        for jj in range(num_subspaces):
            diff_sum = 0
            add = True
            for kk in range(num_vars):
                diff = subspace_indices[kk][jj] - subspace_indices[kk][ii]
                if diff > 1 or diff < 0:
                    add = False
                    break
                diff_sum += diff
            if add:
                out[ii] += -1.0 if (diff_sum % 2) else 1.0
    return out
