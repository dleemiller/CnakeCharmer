"""Test midpoint_integrate equivalence."""

import pytest

from cnake_data.cy.numerical.midpoint_integrate import (
    midpoint_integrate as cy_func,
)
from cnake_data.py.numerical.midpoint_integrate import (
    midpoint_integrate as py_func,
)


@pytest.mark.parametrize("start,stop,n", [(0.0, 1.0, 100), (0.0, 5.0, 1000), (1.0, 10.0, 5000)])
def test_midpoint_integrate_equivalence(start, stop, n):
    py_result = py_func(start, stop, n)
    cy_result = cy_func(start, stop, n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
