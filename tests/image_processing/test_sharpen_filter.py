"""Test sharpen_filter equivalence."""

import pytest

from cnake_charmer.cy.image_processing.sharpen_filter import sharpen_filter as cy_func
from cnake_charmer.py.image_processing.sharpen_filter import sharpen_filter as py_func


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_sharpen_filter_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple)
    assert isinstance(cy_result, tuple)
    assert py_result[0] == cy_result[0], (
        f"Sum mismatch at n={n}: py={py_result[0]}, cy={cy_result[0]}"
    )
    assert py_result[1] == cy_result[1], (
        f"Clipped count mismatch at n={n}: py={py_result[1]}, cy={cy_result[1]}"
    )
