"""Test level_set_delta equivalence."""

import pytest

from cnake_data.cy.numerical.level_set_delta import level_set_delta as cy_func
from cnake_data.py.numerical.level_set_delta import level_set_delta as py_func


@pytest.mark.parametrize("n", [20, 50, 100, 200])
def test_level_set_delta_equivalence(n):
    total_py, max_py, count_py = py_func(n)
    total_cy, max_cy, count_cy = cy_func(n)
    assert count_py == count_cy
    assert abs(total_py - total_cy) / max(abs(total_py), 1.0) < 1e-9
    assert abs(max_py - max_cy) / max(abs(max_py), 1.0) < 1e-9
