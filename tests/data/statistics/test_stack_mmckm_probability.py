"""Test stack_mmckm_probability equivalence."""

import pytest

from cnake_data.cy.statistics.stack_mmckm_probability import stack_mmckm_probability as cy_func
from cnake_data.py.statistics.stack_mmckm_probability import stack_mmckm_probability as py_func


@pytest.mark.parametrize(
    "args", [(30, 20, 6, 4, 12), (45, 28, 9, 6, 20), (60, 35, 8, 7, 24), (80, 50, 12, 9, 30)]
)
def test_stack_mmckm_probability_equivalence(args):
    assert py_func(*args) == cy_func(*args)
