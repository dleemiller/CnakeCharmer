"""Test stack_fixpoint_pipeline equivalence."""

import pytest

from cnake_data.cy.numerical.stack_fixpoint_pipeline import stack_fixpoint_pipeline as cy_func
from cnake_data.py.numerical.stack_fixpoint_pipeline import stack_fixpoint_pipeline as py_func


@pytest.mark.parametrize("n", [100, 500, 2000, 8000])
def test_stack_fixpoint_pipeline_equivalence(n):
    assert py_func(n) == cy_func(n)
