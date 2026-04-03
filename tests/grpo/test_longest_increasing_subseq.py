"""Test longest_increasing_subseq."""

import pytest

from cnake_charmer.py.grpo.longest_increasing_subseq import longest_increasing_subseq


@pytest.mark.parametrize("n", [100, 1000, 5000])
def test_longest_increasing_subseq(n):
    assert longest_increasing_subseq(n) == longest_increasing_subseq(n)
