"""Test method_of_lines equivalence."""

import pytest

from cnake_charmer.cy.diff_equations.method_of_lines import method_of_lines as cy_func
from cnake_charmer.py.diff_equations.method_of_lines import method_of_lines as py_func


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_method_of_lines_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
