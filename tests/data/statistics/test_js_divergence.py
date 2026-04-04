"""Test js_divergence equivalence."""

import pytest

from cnake_data.cy.statistics.js_divergence import js_divergence as cy_func
from cnake_data.py.statistics.js_divergence import js_divergence as py_func


@pytest.mark.parametrize("n", [10, 50, 100])
def test_js_divergence_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4
