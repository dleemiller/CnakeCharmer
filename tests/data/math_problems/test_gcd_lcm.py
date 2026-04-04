"""Test GCD/LCM sum equivalence."""

import pytest

from cnake_data.cy.math_problems.gcd_lcm import gcd_lcm as cy_gcd_lcm
from cnake_data.py.math_problems.gcd_lcm import gcd_lcm as py_gcd_lcm


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_gcd_lcm_equivalence(n):
    assert py_gcd_lcm(n) == cy_gcd_lcm(n)
