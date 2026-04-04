"""Test harris_corner equivalence."""

import pytest

from cnake_data.cy.image_processing.harris_corner import harris_corner as cy_func
from cnake_data.py.image_processing.harris_corner import harris_corner as py_func


@pytest.mark.parametrize("n", [32, 64, 100, 150])
def test_harris_corner_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # num_corners is integer
    assert py_result[0] == cy_result[0], (
        f"Corner count mismatch: py={py_result[0]}, cy={cy_result[0]}"
    )
    # Float comparisons
    for p, c in zip(py_result[1:], cy_result[1:], strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
