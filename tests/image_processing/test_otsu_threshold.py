"""Test otsu_threshold equivalence."""

import pytest

from cnake_charmer.cy.image_processing.otsu_threshold import otsu_threshold as cy_func
from cnake_charmer.py.image_processing.otsu_threshold import otsu_threshold as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 300])
def test_otsu_threshold_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # Threshold and foreground count must match exactly (integer values)
    assert py_result[0] == cy_result[0], f"Threshold mismatch: py={py_result[0]}, cy={cy_result[0]}"
    assert py_result[1] == cy_result[1], (
        f"Foreground mismatch: py={py_result[1]}, cy={cy_result[1]}"
    )
    # Variance is float
    assert abs(py_result[2] - cy_result[2]) / max(abs(py_result[2]), 1.0) < 1e-4, (
        f"Variance mismatch: py={py_result[2]}, cy={cy_result[2]}"
    )
