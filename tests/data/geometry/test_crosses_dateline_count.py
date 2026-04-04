"""Test crosses_dateline_count equivalence."""

import pytest

from cnake_data.cy.geometry.crosses_dateline_count import crosses_dateline_count as cy_func
from cnake_data.py.geometry.crosses_dateline_count import crosses_dateline_count as py_func


@pytest.mark.parametrize("seed,segment_count", [(7, 100), (9, 1000), (123, 5000)])
def test_crosses_dateline_count_equivalence(seed, segment_count):
    assert py_func(seed, segment_count) == cy_func(seed, segment_count)
