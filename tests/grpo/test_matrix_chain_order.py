"""Test matrix_chain_order."""

import pytest

from cnake_charmer.py.grpo.matrix_chain_order import matrix_chain_order


@pytest.mark.parametrize("n", [3, 10, 30, 50])
def test_matrix_chain_order(n):
    assert matrix_chain_order(n) == matrix_chain_order(n)
