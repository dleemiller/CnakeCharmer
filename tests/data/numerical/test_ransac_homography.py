"""Test ransac_homography equivalence."""

import pytest

from cnake_data.cy.numerical.ransac_homography import ransac_homography as cy_func
from cnake_data.py.numerical.ransac_homography import ransac_homography as py_func


@pytest.mark.parametrize(
    "n_pts,n_iter",
    [
        (100, 40),
        (200, 80),
        (150, 60),
        (80, 30),
    ],
)
def test_ransac_homography_equivalence(n_pts, n_iter):
    assert py_func(n_pts, n_iter) == cy_func(n_pts, n_iter)
