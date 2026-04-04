"""Test memcpy_block_reverse equivalence."""

import pytest

from cnake_data.cy.algorithms.memcpy_block_reverse import (
    memcpy_block_reverse as cy_func,
)
from cnake_data.py.algorithms.memcpy_block_reverse import (
    memcpy_block_reverse as py_func,
)


@pytest.mark.parametrize("n", [128, 1024, 10000, 50000])
def test_memcpy_block_reverse_equivalence(n):
    assert py_func(n) == cy_func(n)
