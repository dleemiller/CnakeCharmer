"""Test anon_enum_flags equivalence."""

import pytest

from cnake_charmer.cy.algorithms.anon_enum_flags import anon_enum_flags as cy_func
from cnake_charmer.py.algorithms.anon_enum_flags import anon_enum_flags as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_anon_enum_flags_equivalence(n):
    assert py_func(n) == cy_func(n)
