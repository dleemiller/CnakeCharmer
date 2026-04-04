"""Test nogil_newton_roots equivalence."""

import pytest

from cnake_data.cy.numerical.nogil_newton_roots import (
    nogil_newton_roots as cy_func,
)
from cnake_data.py.numerical.nogil_newton_roots import (
    nogil_newton_roots as py_func,
)


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_nogil_newton_roots_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    rel = abs(py_result - cy_result) / max(abs(py_result), 1.0)
    assert rel < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
