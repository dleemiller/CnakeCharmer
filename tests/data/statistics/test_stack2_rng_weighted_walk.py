import pytest

from cnake_data.cy.statistics.stack2_rng_weighted_walk import stack2_rng_weighted_walk as cy_func
from cnake_data.py.statistics.stack2_rng_weighted_walk import stack2_rng_weighted_walk as py_func


@pytest.mark.parametrize(
    "bucket_count, draw_count, seed_offset",
    [(8, 5000, 3), (16, 20000, 11), (32, 60000, 17)],
)
def test_stack2_rng_weighted_walk_equivalence(bucket_count, draw_count, seed_offset):
    assert py_func(bucket_count, draw_count, seed_offset) == cy_func(
        bucket_count, draw_count, seed_offset
    )
