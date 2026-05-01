from __future__ import annotations


def update_smolyak_coefficients(
    new_index: list[int], subspace_indices: list[list[int]], smolyak_coeffs: list[float]
) -> list[float]:
    num_vars = len(subspace_indices)
    if num_vars == 0:
        return smolyak_coeffs
    num_subspace = len(subspace_indices[0])

    for ii in range(num_subspace):
        diff_sum = 0
        update = True
        for jj in range(num_vars):
            diff = new_index[jj] - subspace_indices[jj][ii]
            diff_sum += diff
            if diff < 0 or diff > 1:
                update = False
                break
        if update:
            smolyak_coeffs[ii] += -1.0 if (diff_sum % 2) else 1.0
    return smolyak_coeffs
