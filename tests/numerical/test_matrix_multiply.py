"""Test matrix multiplication equivalence."""

import pytest

from cnake_charmer.cy.numerical.matrix_multiply import matrix_multiply as cy_matrix_multiply
from cnake_charmer.py.numerical.matrix_multiply import matrix_multiply as py_matrix_multiply


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_matrix_multiply_equivalence(n):
    py_result = py_matrix_multiply(n)
    cy_result = cy_matrix_multiply(n)
    for i in range(n):
        for j in range(n):
            assert abs(py_result[i][j] - cy_result[i][j]) < 1e-6, f"Mismatch at [{i}][{j}]"
