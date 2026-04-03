"""Test eigenvalues_3x3 equivalence."""

import pytest

from cnake_charmer.cy.numerical.eigenvalues_3x3 import eigenvalues_3x3 as cy_func
from cnake_charmer.py.numerical.eigenvalues_3x3 import eigenvalues_3x3 as py_func


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_eigenvalues_3x3_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
