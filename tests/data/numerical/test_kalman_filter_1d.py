"""Test kalman_filter_1d equivalence."""

import pytest

from cnake_data.cy.numerical.kalman_filter_1d import kalman_filter_1d as cy_func
from cnake_data.py.numerical.kalman_filter_1d import kalman_filter_1d as py_func


@pytest.mark.parametrize(
    "n_steps",
    [500, 1000, 2000, 5000],
)
def test_kalman_filter_1d_equivalence(n_steps):
    assert py_func(n_steps) == cy_func(n_steps)
