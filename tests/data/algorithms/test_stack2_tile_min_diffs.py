import pytest

from cnake_data.cy.algorithms.stack2_tile_min_diffs import stack2_tile_min_diffs as cy_func
from cnake_data.py.algorithms.stack2_tile_min_diffs import stack2_tile_min_diffs as py_func


@pytest.mark.parametrize(
    "tile_rows, tile_cols, tile_count, sample_count",
    [(8, 8, 8, 6), (12, 10, 12, 10), (16, 16, 18, 14)],
)
def test_stack2_tile_min_diffs_equivalence(tile_rows, tile_cols, tile_count, sample_count):
    assert py_func(tile_rows, tile_cols, tile_count, sample_count) == cy_func(
        tile_rows, tile_cols, tile_count, sample_count
    )
