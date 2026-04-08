"""Test rbf_warp equivalence."""

import pytest

from cnake_data.cy.image_processing.rbf_warp import rbf_warp as cy_func
from cnake_data.py.image_processing.rbf_warp import rbf_warp as py_func


@pytest.mark.parametrize(
    "rows,cols,n_ctrl,sigma",
    [
        (30, 30, 9, 15.0),
        (60, 60, 12, 20.0),
        (40, 40, 9, 10.0),
        (50, 50, 16, 25.0),
    ],
)
def test_rbf_warp_equivalence(rows, cols, n_ctrl, sigma):
    assert py_func(rows, cols, n_ctrl, sigma) == cy_func(rows, cols, n_ctrl, sigma)
