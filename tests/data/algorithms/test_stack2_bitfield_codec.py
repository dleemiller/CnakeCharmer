import pytest

from cnake_data.cy.algorithms.stack2_bitfield_codec import stack2_bitfield_codec as cy_func
from cnake_data.py.algorithms.stack2_bitfield_codec import stack2_bitfield_codec as py_func


@pytest.mark.parametrize(
    "col_count, row_count, seed_tag, trim_bits",
    [(4, 6, 3, 3), (6, 8, 19, 3), (5, 10, 11, 4)],
)
def test_stack2_bitfield_codec_equivalence(col_count, row_count, seed_tag, trim_bits):
    assert py_func(col_count, row_count, seed_tag, trim_bits) == cy_func(
        col_count, row_count, seed_tag, trim_bits
    )
