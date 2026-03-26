"""Test box_blur equivalence."""

import pytest

from cnake_charmer.cy.image_processing.box_blur import box_blur as cy_box_blur
from cnake_charmer.py.image_processing.box_blur import box_blur as py_box_blur


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_box_blur_equivalence(n):
    py_result = py_box_blur(n)
    cy_result = cy_box_blur(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
