"""Test threadlocal_counter_class equivalence."""

import pytest

from cnake_data.cy.algorithms.threadlocal_counter_class import (
    threadlocal_counter_class as cy_func,
)
from cnake_data.py.algorithms.threadlocal_counter_class import (
    threadlocal_counter_class as py_func,
)


@pytest.mark.parametrize(
    "nslots,steps,seed,stride", [(16, 300, 5, 3), (32, 700, 11, 5), (20, 650, 9, 7)]
)
def test_threadlocal_counter_class_equivalence(nslots, steps, seed, stride):
    assert py_func(nslots, steps, seed, stride) == cy_func(nslots, steps, seed, stride)
