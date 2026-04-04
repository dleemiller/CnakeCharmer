"""Test rotating_calipers equivalence."""

import pytest

from cnake_data.cy.geometry.rotating_calipers import rotating_calipers as cy_func
from cnake_data.py.geometry.rotating_calipers import rotating_calipers as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 5000])
def test_rotating_calipers_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
