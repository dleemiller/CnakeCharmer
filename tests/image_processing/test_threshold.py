"""Test threshold equivalence."""

import pytest

from cnake_charmer.cy.image_processing.threshold import threshold as cy_func
from cnake_charmer.py.image_processing.threshold import threshold as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_threshold_equivalence(n):
    assert py_func(n) == cy_func(n)
