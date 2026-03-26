"""Test ising_model equivalence."""

import pytest

from cnake_charmer.cy.simulation.ising_model import ising_model as cy_func
from cnake_charmer.py.simulation.ising_model import ising_model as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_ising_model_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
