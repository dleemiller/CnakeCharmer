"""Test partition_equal_sum equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.partition_equal_sum import (
    partition_equal_sum as cy_partition_equal_sum,
)
from cnake_data.py.dynamic_programming.partition_equal_sum import (
    partition_equal_sum as py_partition_equal_sum,
)


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_partition_equal_sum_equivalence(n):
    assert py_partition_equal_sum(n) == cy_partition_equal_sum(n)
