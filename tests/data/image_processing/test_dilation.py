"""Test dilation equivalence."""

import pytest

from cnake_data.cy.image_processing.dilation import dilation as cy_func
from cnake_data.py.image_processing.dilation import dilation as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_dilation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
