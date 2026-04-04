"""Test bilinear_interpolation equivalence."""

import pytest

from cnake_data.cy.image_processing.bilinear_interpolation import (
    bilinear_interpolation as cy_func,
)
from cnake_data.py.image_processing.bilinear_interpolation import (
    bilinear_interpolation as py_func,
)


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_bilinear_interpolation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
