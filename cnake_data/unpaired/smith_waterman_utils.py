"""Matrix generation and backtrack helpers for local alignment."""

from __future__ import annotations

import numpy as np

SOFT_CLIPPED = -2
GAP = -1


def generate_matrices(
    query,
    target,
    match_bonus,
    mismatch_penalty,
    indel_penalty,
    force_query_start,
    force_target_start,
    force_either_start,
):
    shape = (len(query) + 1, len(target) + 1)
    scores = np.zeros(shape, int)
    row_directions = np.zeros(shape, int)
    col_directions = np.zeros(shape, int)

    if force_query_start:
        for row in range(1, len(query) + 1):
            scores[row, 0] = scores[row - 1, 0] + indel_penalty
            row_directions[row, 0] = -1

    if force_target_start:
        for col in range(1, len(target) + 1):
            scores[0, col] = scores[0, col - 1] + indel_penalty
            col_directions[0, col] = -1

    unconstrained_start = not (force_query_start or force_target_start or force_either_start)

    max_score = 0
    max_row = 0
    max_col = 0

    for row in range(1, len(query) + 1):
        for col in range(1, len(target) + 1):
            if query[row - 1] == "N" or target[col - 1] == "N" or query[row - 1] == target[col - 1]:
                match_or_mismatch = match_bonus
            else:
                match_or_mismatch = mismatch_penalty

            diagonal = scores[row - 1, col - 1] + match_or_mismatch
            from_left = scores[row, col - 1] + indel_penalty
            from_above = scores[row - 1, col] + indel_penalty
            new_score = max(diagonal, from_left, from_above)
            if unconstrained_start:
                new_score = max(0, new_score)
            scores[row, col] = new_score
            if new_score > max_score:
                max_score = new_score
                max_row = row
                max_col = col

            if unconstrained_start and new_score == 0:
                pass
            elif new_score == diagonal:
                col_directions[row, col] = -1
                row_directions[row, col] = -1
            elif new_score == from_left:
                col_directions[row, col] = -1
            elif new_score == from_above:
                row_directions[row, col] = -1

    return {
        "scores": scores,
        "row_directions": row_directions,
        "col_directions": col_directions,
        "max_row": max_row,
        "max_col": max_col,
        "max_score": max_score,
    }
