"""Test bilinear_interpolation equivalence."""

import pytest

from cnake_charmer.cy.image_processing.bilinear_interpolation import (
    bilinear_interpolation as cy_func,
)
from cnake_charmer.py.image_processing.bilinear_interpolation import (
    bilinear_interpolation as py_func,
)


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_bilinear_interpolation_equivalence(n):
    assert py_func(n) == cy_func(n)
