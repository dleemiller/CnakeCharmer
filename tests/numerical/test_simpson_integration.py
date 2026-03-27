"""Test simpson_integration equivalence."""

import pytest

from cnake_charmer.cy.numerical.simpson_integration import simpson_integration as cy_func
from cnake_charmer.py.numerical.simpson_integration import simpson_integration as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_simpson_integration_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # Third element is int (num_panels)
    assert py_result[2] == cy_result[2]
    # Float elements
    for i in range(2):
        rel = abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0)
        assert rel < 1e-4, f"Mismatch at element {i}: {py_result[i]} vs {cy_result[i]}"
