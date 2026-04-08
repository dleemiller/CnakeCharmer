"""Test param_grid_eval equivalence."""

import pytest

from cnake_data.cy.numerical.param_grid_eval import param_grid_eval as cy_func
from cnake_data.py.numerical.param_grid_eval import param_grid_eval as py_func


@pytest.mark.parametrize(
    "n_h,n_xyz",
    [
        (4, 4),
        (6, 6),
        (8, 8),
        (12, 10),
    ],
)
def test_param_grid_eval_equivalence(n_h, n_xyz):
    assert py_func(n_h, n_xyz) == cy_func(n_h, n_xyz)
