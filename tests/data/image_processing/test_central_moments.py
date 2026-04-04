"""Test central_moments equivalence."""

import pytest

from cnake_data.cy.image_processing.central_moments import central_moments as cy_func
from cnake_data.py.image_processing.central_moments import central_moments as py_func


@pytest.mark.parametrize("n", [16, 40, 80, 120])
def test_central_moments_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i, (p, c) in enumerate(zip(py_result, cy_result, strict=False)):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch at index {i}: py={p}, cy={c}"
