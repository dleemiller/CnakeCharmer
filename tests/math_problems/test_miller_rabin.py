"""Test miller_rabin equivalence."""

import pytest

from cnake_charmer.cy.math_problems.miller_rabin import miller_rabin as cy_func
from cnake_charmer.py.math_problems.miller_rabin import miller_rabin as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000, 5000])
def test_miller_rabin_equivalence(n):
    assert py_func(n) == cy_func(n)
