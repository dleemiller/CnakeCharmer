"""Test matrix_multiply equivalence."""

import pytest

from cnake_charmer.cy.numerical.matrix_multiply import matrix_multiply as cy_func
from cnake_charmer.py.numerical.matrix_multiply import matrix_multiply as py_func


@pytest.mark.parametrize("n", [2, 5, 10, 50])
def test_matrix_multiply_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i in range(3):
        rel = abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0)
        assert rel < 1e-4, f"Mismatch at element {i}: {py_result[i]} vs {cy_result[i]}"
