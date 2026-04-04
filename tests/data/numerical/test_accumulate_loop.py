"""Test accumulate_loop equivalence."""

import pytest

from cnake_data.cy.numerical.accumulate_loop import (
    accumulate_divisions as cy_accumulate_divisions,
)
from cnake_data.py.numerical.accumulate_loop import (
    accumulate_divisions as py_accumulate_divisions,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_accumulate_divisions_equivalence(n):
    assert abs(py_accumulate_divisions(n) - cy_accumulate_divisions(n)) < 1e-6
