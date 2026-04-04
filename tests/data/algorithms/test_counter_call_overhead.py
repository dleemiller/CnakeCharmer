"""Test counter_call_overhead equivalence."""

import pytest

from cnake_data.cy.algorithms.counter_call_overhead import counter_call_overhead as cy_func
from cnake_data.py.algorithms.counter_call_overhead import counter_call_overhead as py_func


@pytest.mark.parametrize(
    "limit,repeats,offset",
    [
        (10, 1, 0),
        (100, 3, 2),
        (2000, 2, -1),
        (5000, 4, 7),
    ],
)
def test_counter_call_overhead_equivalence(limit, repeats, offset):
    assert py_func(limit, repeats, offset) == cy_func(limit, repeats, offset)
