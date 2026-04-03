"""Test jaro_winkler equivalence."""

import pytest

from cnake_charmer.cy.string_processing.jaro_winkler import jaro_winkler as cy_func
from cnake_charmer.py.string_processing.jaro_winkler import jaro_winkler as py_func


@pytest.mark.parametrize("n", [10, 50, 80])
def test_jaro_winkler_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
