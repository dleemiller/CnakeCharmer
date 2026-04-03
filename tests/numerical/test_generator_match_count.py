"""Test generator_match_count equivalence."""

import pytest

from cnake_charmer.cy.numerical.generator_match_count import generator_match_count as cy_gen_match
from cnake_charmer.py.numerical.generator_match_count import generator_match_count as py_gen_match


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_generator_match_count_equivalence(n):
    assert py_gen_match(n) == cy_gen_match(n)
