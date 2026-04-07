"""Test flow_shop_makespan equivalence."""

import pytest

from cnake_data.cy.optimization.flow_shop_makespan import flow_shop_makespan as cy_func
from cnake_data.py.optimization.flow_shop_makespan import flow_shop_makespan as py_func


@pytest.mark.parametrize(
    "n,m",
    [(3, 2), (5, 3), (10, 4), (40, 8)],
)
def test_flow_shop_makespan_equivalence(n, m):
    py_val = py_func(n, m)
    cy_val = cy_func(n, m)
    assert abs(py_val - cy_val) / max(abs(py_val), 1.0) < 1e-9
