"""Test window_functions equivalence."""

import pytest

from cnake_data.cy.dsp.window_functions import window_functions as cy_window_functions
from cnake_data.py.dsp.window_functions import window_functions as py_window_functions


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_window_functions_equivalence(n):
    py_result = py_window_functions(n)
    cy_result = cy_window_functions(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
