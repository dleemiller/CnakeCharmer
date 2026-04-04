"""Test bounded_mine_prob equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.bounded_mine_prob import bounded_mine_prob as cy_func
from cnake_data.py.dynamic_programming.bounded_mine_prob import bounded_mine_prob as py_func


@pytest.mark.parametrize(
    "num_cells,total_mines,max_per_cell,repeats",
    [
        (8, 10, 2, 3),
        (12, 16, 3, 4),
        (14, 20, 4, 5),
    ],
)
def test_bounded_mine_prob_equivalence(num_cells, total_mines, max_per_cell, repeats):
    py_result = py_func(num_cells, total_mines, max_per_cell, repeats)
    cy_result = cy_func(num_cells, total_mines, max_per_cell, repeats)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
