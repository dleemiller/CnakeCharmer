"""Test lempel_ziv_complexity equivalence."""

import pytest

from cnake_charmer.cy.algorithms.lempel_ziv_complexity import lempel_ziv_complexity as cy_func
from cnake_charmer.py.algorithms.lempel_ziv_complexity import lempel_ziv_complexity as py_func


@pytest.mark.parametrize("n", [10, 100, 500])
def test_lempel_ziv_complexity_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4
