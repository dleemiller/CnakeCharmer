"""Test sudoku_solver equivalence."""

import pytest

from cnake_data.cy.algorithms.sudoku_solver import sudoku_solver as cy_sudoku_solver
from cnake_data.py.algorithms.sudoku_solver import sudoku_solver as py_sudoku_solver


@pytest.mark.parametrize("n", [1, 5, 10, 20])
def test_sudoku_solver_equivalence(n):
    py_result = py_sudoku_solver(n)
    cy_result = cy_sudoku_solver(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
