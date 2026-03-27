"""Test anon_enum_limits equivalence."""

import pytest

from cnake_charmer.cy.numerical.anon_enum_limits import anon_enum_limits as cy_func
from cnake_charmer.py.numerical.anon_enum_limits import anon_enum_limits as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_anon_enum_limits_equivalence(n):
    assert py_func(n) == cy_func(n)
