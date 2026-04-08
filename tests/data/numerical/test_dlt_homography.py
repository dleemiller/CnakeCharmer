"""Test dlt_homography equivalence."""

import pytest

from cnake_data.cy.numerical.dlt_homography import dlt_homography as cy_func
from cnake_data.py.numerical.dlt_homography import dlt_homography as py_func


@pytest.mark.parametrize(
    "n_pts",
    [250, 500, 1000, 2000],
)
def test_dlt_homography_equivalence(n_pts):
    assert py_func(n_pts) == cy_func(n_pts)
