"""Test ufunc_classify_bin equivalence."""

import pytest

from cnake_charmer.cy.statistics.ufunc_classify_bin import (
    ufunc_classify_bin as cy_func,
)
from cnake_charmer.py.statistics.ufunc_classify_bin import (
    ufunc_classify_bin as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 100000])
def test_ufunc_classify_bin_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
