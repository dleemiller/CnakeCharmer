"""Test vec3_class_cross_norm equivalence."""

import pytest

from cnake_data.cy.geometry.vec3_class_cross_norm import vec3_class_cross_norm as cy_func
from cnake_data.py.geometry.vec3_class_cross_norm import vec3_class_cross_norm as py_func


@pytest.mark.parametrize("steps,bias", [(150, 0.1), (220, -0.05), (300, 0.2)])
def test_vec3_class_cross_norm_equivalence(steps, bias):
    py_result = py_func(steps, bias)
    cy_result = cy_func(steps, bias)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
