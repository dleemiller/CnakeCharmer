"""Test stack_ece_segments equivalence."""

import pytest

from cnake_data.cy.algorithms.stack_ece_segments import stack_ece_segments as cy_func
from cnake_data.py.algorithms.stack_ece_segments import stack_ece_segments as py_func


@pytest.mark.parametrize("args", [(120, 8), (240, 12), (600, 18), (1200, 24)])
def test_stack_ece_segments_equivalence(args):
    assert py_func(*args) == cy_func(*args)
