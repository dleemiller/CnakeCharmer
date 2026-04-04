"""Test gauss_seidel equivalence."""

import pytest

from cnake_data.cy.numerical.gauss_seidel import gauss_seidel as cy_func
from cnake_data.py.numerical.gauss_seidel import gauss_seidel as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_gauss_seidel_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i in range(3):
        rel = abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0)
        assert rel < 1e-4, f"Mismatch at element {i}: {py_result[i]} vs {cy_result[i]}"
