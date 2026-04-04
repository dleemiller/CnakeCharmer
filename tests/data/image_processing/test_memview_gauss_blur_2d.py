"""Test 2D Gaussian blur equivalence."""

import pytest

from cnake_data.cy.image_processing.memview_gauss_blur_2d import (
    memview_gauss_blur_2d as cy_memview_gauss_blur_2d,
)
from cnake_data.py.image_processing.memview_gauss_blur_2d import (
    memview_gauss_blur_2d as py_memview_gauss_blur_2d,
)


@pytest.mark.parametrize("n", [10, 50, 100, 300])
def test_memview_gauss_blur_2d_equivalence(n):
    py_result = py_memview_gauss_blur_2d(n)
    cy_result = cy_memview_gauss_blur_2d(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
