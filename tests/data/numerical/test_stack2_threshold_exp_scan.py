import pytest

from cnake_data.cy.numerical.stack2_threshold_exp_scan import stack2_threshold_exp_scan as cy_func
from cnake_data.py.numerical.stack2_threshold_exp_scan import stack2_threshold_exp_scan as py_func


@pytest.mark.parametrize(
    "vector_size, threshold_milli, seed_tag",
    [(50000, 550, 3), (100000, 620, 9), (180000, 700, 17)],
)
def test_stack2_threshold_exp_scan_equivalence(vector_size, threshold_milli, seed_tag):
    assert py_func(vector_size, threshold_milli, seed_tag) == cy_func(
        vector_size, threshold_milli, seed_tag
    )
