"""Test sparse_dot_product."""

import pytest

from cnake_charmer.py.grpo.sparse_dot_product import sparse_dot_product


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_sparse_dot_product(n):
    result = sparse_dot_product(n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == sparse_dot_product(n)
