"""Test great circle distance equivalence."""

import pytest

from cnake_data.cy.numerical.great_circle import great_circle as cy_great_circle
from cnake_data.py.numerical.great_circle import great_circle as py_great_circle


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_great_circle_equivalence(n):
    py_result = py_great_circle(n)
    cy_result = cy_great_circle(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
