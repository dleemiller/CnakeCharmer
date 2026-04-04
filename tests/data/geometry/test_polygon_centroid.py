"""Test polygon_centroid equivalence."""

import pytest

from cnake_data.cy.geometry.polygon_centroid import polygon_centroid as cy_func
from cnake_data.py.geometry.polygon_centroid import polygon_centroid as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_polygon_centroid_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple)
    assert isinstance(cy_result, tuple)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={p}, cy={c}"
