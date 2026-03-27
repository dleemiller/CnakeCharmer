"""Test nogil_nbody_step equivalence."""

import pytest

from cnake_charmer.cy.simulation.nogil_nbody_step import (
    nogil_nbody_step as cy_func,
)
from cnake_charmer.py.simulation.nogil_nbody_step import (
    nogil_nbody_step as py_func,
)


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_nogil_nbody_step_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    rel = abs(py_result - cy_result) / max(abs(py_result), 1e-15)
    assert rel < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
