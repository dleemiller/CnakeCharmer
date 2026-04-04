"""Test traffic_flow equivalence."""

import pytest

from cnake_data.cy.simulation.traffic_flow import traffic_flow as cy_func
from cnake_data.py.simulation.traffic_flow import traffic_flow as py_func


@pytest.mark.parametrize("n", [30, 100, 300, 600])
def test_traffic_flow_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
