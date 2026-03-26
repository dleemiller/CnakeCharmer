"""Test cubic_spline_eval equivalence."""

import pytest

from cnake_charmer.cy.numerical.cubic_spline_eval import (
    cubic_spline_eval as cy_cubic_spline_eval,
)
from cnake_charmer.py.numerical.cubic_spline_eval import (
    cubic_spline_eval as py_cubic_spline_eval,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_cubic_spline_eval_equivalence(n):
    py_result = py_cubic_spline_eval(n)
    cy_result = cy_cubic_spline_eval(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
