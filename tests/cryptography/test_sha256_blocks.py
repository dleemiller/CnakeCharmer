"""Test sha256_blocks equivalence."""

import pytest

from cnake_charmer.cy.cryptography.sha256_blocks import sha256_blocks as cy_func
from cnake_charmer.py.cryptography.sha256_blocks import sha256_blocks as py_func


@pytest.mark.parametrize("n", [64, 128, 512, 1024])
def test_sha256_blocks_equivalence(n):
    assert py_func(n) == cy_func(n)
