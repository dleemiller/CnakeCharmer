"""Test erosion equivalence."""

import pytest

from cnake_charmer.cy.image_processing.erosion import erosion as cy_func
from cnake_charmer.py.image_processing.erosion import erosion as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_erosion_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
