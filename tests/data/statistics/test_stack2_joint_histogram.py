import pytest

from cnake_data.cy.statistics.stack2_joint_histogram import stack2_joint_histogram as cy_func
from cnake_data.py.statistics.stack2_joint_histogram import stack2_joint_histogram as py_func


@pytest.mark.parametrize(
    "sample_count, bin_count, scale_tag",
    [(6000, 16, 3), (12000, 24, 9), (30000, 32, 17)],
)
def test_stack2_joint_histogram_equivalence(sample_count, bin_count, scale_tag):
    assert py_func(sample_count, bin_count, scale_tag) == cy_func(
        sample_count, bin_count, scale_tag
    )
