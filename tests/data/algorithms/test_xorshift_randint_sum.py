"""Test xorshift_randint_sum equivalence."""

import pytest

from cnake_data.cy.algorithms.xorshift_randint_sum import xorshift_randint_sum as cy_func
from cnake_data.py.algorithms.xorshift_randint_sum import xorshift_randint_sum as py_func


@pytest.mark.parametrize(
    "seed,draws,bucket",
    [(2463534242, 10, 1000), (1, 1000, 97), (123456789, 10000, 251)],
)
def test_xorshift_randint_sum_equivalence(seed, draws, bucket):
    assert py_func(seed, draws, bucket) == cy_func(seed, draws, bucket)
