"""Test stack_prime_bubble_combo equivalence."""

import pytest

from cnake_data.cy.algorithms.stack_prime_bubble_combo import stack_prime_bubble_combo as cy_func
from cnake_data.py.algorithms.stack_prime_bubble_combo import stack_prime_bubble_combo as py_func


@pytest.mark.parametrize("args", [(200, 32), (500, 64), (1000, 128), (1800, 220)])
def test_stack_prime_bubble_combo_equivalence(args):
    assert py_func(*args) == cy_func(*args)
