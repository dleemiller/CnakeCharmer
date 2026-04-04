"""Test central_moments equivalence."""

import pytest

from cnake_charmer.cy.image_processing.central_moments import central_moments as cy_func
from cnake_charmer.py.image_processing.central_moments import central_moments as py_func


@pytest.mark.parametrize("n", [50, 100, 200, 300])
def test_central_moments_equivalence(n):
    assert py_func(n) == cy_func(n)
