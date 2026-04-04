"""Test packed_struct_pixel equivalence."""

import pytest

from cnake_data.cy.image_processing.packed_struct_pixel import (
    packed_struct_pixel as cy_func,
)
from cnake_data.py.image_processing.packed_struct_pixel import (
    packed_struct_pixel as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_packed_struct_pixel_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
