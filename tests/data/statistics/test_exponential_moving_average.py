"""Test exponential_moving_average equivalence."""

import pytest

from cnake_data.cy.statistics.exponential_moving_average import (
    exponential_moving_average as cy_func,
)
from cnake_data.py.statistics.exponential_moving_average import (
    exponential_moving_average as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 5000, 20000])
def test_exponential_moving_average_equivalence(n):
    py = py_func(n)
    cy = cy_func(n)
    assert len(py) == len(cy)
    for a, b in zip(py, cy, strict=False):
        assert abs(a - b) / max(abs(a), 1e-10) < 1e-10
