"""Sudoku grid generation utilities from base-row permutations."""

from __future__ import annotations

from itertools import permutations

MOVE_WAY = (
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (0, 6, 3, 1, 4, 7, 2, 5, 8),
    (0, 3, 6, 4, 1, 7, 2, 5, 8),
    (0, 6, 3, 4, 1, 7, 2, 5, 8),
)


def first_rows_with_leading_digit(leading=5):
    digits = [d for d in range(1, 10) if d != leading]
    for perm in permutations(digits):
        yield [leading, *perm]


def make_grid_from_first_row(first_row, move_pattern):
    return [
        [first_row[idx] for idx in row_order] for row_order in _row_order_from_pattern(move_pattern)
    ]


def _row_order_from_pattern(pattern):
    bands = [pattern[0:3], pattern[3:6], pattern[6:9]]
    return [
        bands[0],
        bands[1],
        bands[2],
        bands[1],
        bands[2],
        bands[0],
        bands[2],
        bands[0],
        bands[1],
    ]
