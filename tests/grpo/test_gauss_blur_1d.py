"""Test gauss_blur_1d."""

import pytest

from cnake_charmer.py.grpo.gauss_blur_1d import gauss_blur_1d


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_gauss_blur_1d(n):
    assert gauss_blur_1d(n) == gauss_blur_1d(n)
