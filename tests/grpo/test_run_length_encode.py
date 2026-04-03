"""Test run_length_encode."""

import pytest

from cnake_charmer.py.grpo.run_length_encode import run_length_encode


@pytest.mark.parametrize("n", [0, 1, 10, 100, 1000])
def test_run_length_encode(n):
    result = run_length_encode(n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == run_length_encode(n)  # deterministic
