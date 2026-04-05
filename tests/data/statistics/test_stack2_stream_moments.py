import pytest

from cnake_data.cy.statistics.stack2_stream_moments import stack2_stream_moments as cy_func
from cnake_data.py.statistics.stack2_stream_moments import stack2_stream_moments as py_func


@pytest.mark.parametrize(
    "sample_count, shift_tag, stride_tag",
    [(5000, 3, 1), (20000, 9, 5), (60000, 17, 7)],
)
def test_stack2_stream_moments_equivalence(sample_count, shift_tag, stride_tag):
    assert py_func(sample_count, shift_tag, stride_tag) == cy_func(
        sample_count, shift_tag, stride_tag
    )
