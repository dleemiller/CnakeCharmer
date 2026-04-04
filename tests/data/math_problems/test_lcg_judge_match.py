"""Test LCG judge match equivalence."""

import pytest

from cnake_data.cy.math_problems.lcg_judge_match import lcg_judge_match as cy_lcg_judge_match
from cnake_data.py.math_problems.lcg_judge_match import lcg_judge_match as py_lcg_judge_match


@pytest.mark.parametrize("n", [10, 100, 1000, 40000])
def test_lcg_judge_match_equivalence(n):
    py_result = py_lcg_judge_match(n)
    cy_result = cy_lcg_judge_match(n)
    assert py_result == cy_result
