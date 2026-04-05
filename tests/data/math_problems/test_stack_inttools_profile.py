"""Test stack_inttools_profile equivalence."""

import pytest

from cnake_data.cy.math_problems.stack_inttools_profile import stack_inttools_profile as cy_func
from cnake_data.py.math_problems.stack_inttools_profile import stack_inttools_profile as py_func


@pytest.mark.parametrize("n", [200, 800, 1600, 3000])
def test_stack_inttools_profile_equivalence(n):
    assert py_func(n) == cy_func(n)
