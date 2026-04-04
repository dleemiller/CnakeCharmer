"""Test range_loop_checksum equivalence."""

import pytest

from cnake_data.cy.algorithms.range_loop_checksum import range_loop_checksum as cy_func
from cnake_data.py.algorithms.range_loop_checksum import range_loop_checksum as py_func


@pytest.mark.parametrize(
    "a,b,step,factor,rounds",
    [
        (5, 30, 2, 2, 40),
        (12, 80, 3, 3, 120),
        (7, 55, 4, 5, 90),
    ],
)
def test_range_loop_checksum_equivalence(a, b, step, factor, rounds):
    assert py_func(a, b, step, factor, rounds) == cy_func(a, b, step, factor, rounds)
