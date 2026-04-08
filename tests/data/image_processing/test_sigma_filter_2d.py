"""Test sigma_filter_2d equivalence."""

import pytest

from cnake_data.cy.image_processing.sigma_filter_2d import sigma_filter_2d as cy_func
from cnake_data.py.image_processing.sigma_filter_2d import sigma_filter_2d as py_func


@pytest.mark.parametrize(
    "n,m,radius,threshold",
    [
        (10, 12, 2, 0.5),
        (20, 25, 2, 0.3),
        (30, 40, 3, 0.5),
        (60, 80, 3, 0.5),
    ],
)
def test_sigma_filter_2d_equivalence(n, m, radius, threshold):
    assert py_func(n, m, radius, threshold) == cy_func(n, m, radius, threshold)
