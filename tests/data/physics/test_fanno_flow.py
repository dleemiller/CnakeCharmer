"""Test fanno_flow equivalence."""

import pytest

from cnake_data.cy.physics.fanno_flow import fanno_flow as cy_func
from cnake_data.py.physics.fanno_flow import fanno_flow as py_func


@pytest.mark.parametrize("n", [10, 100, 500])
def test_fanno_flow_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4
