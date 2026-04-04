"""Test convolution_2d equivalence."""

import pytest

from cnake_data.cy.image_processing.convolution_2d import convolution_2d as cy_func
from cnake_data.py.image_processing.convolution_2d import convolution_2d as py_func


@pytest.mark.parametrize("n", [10, 30, 50])
def test_convolution_2d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
