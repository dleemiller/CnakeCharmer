"""Test config_lookup equivalence."""

import pytest

from cnake_data.cy.algorithms.config_lookup import config_lookup as cy_func
from cnake_data.py.algorithms.config_lookup import config_lookup as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_config_lookup_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
