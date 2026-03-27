"""Test ctuple_divmod equivalence."""

import pytest

from cnake_charmer.cy.math_problems.ctuple_divmod import ctuple_divmod as cy_func
from cnake_charmer.py.math_problems.ctuple_divmod import ctuple_divmod as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_ctuple_divmod_equivalence(n):
    assert py_func(n) == cy_func(n)
