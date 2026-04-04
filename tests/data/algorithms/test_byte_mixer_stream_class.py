"""Test byte_mixer_stream_class equivalence."""

import pytest

from cnake_data.cy.algorithms.byte_mixer_stream_class import byte_mixer_stream_class as cy_func
from cnake_data.py.algorithms.byte_mixer_stream_class import byte_mixer_stream_class as py_func


@pytest.mark.parametrize(
    "n_bytes,window,rounds,seed,key_scale",
    [(600, 17, 3, 7, 5), (900, 23, 3, 13, 9), (1200, 31, 4, 29, 11)],
)
def test_byte_mixer_stream_class_equivalence(n_bytes, window, rounds, seed, key_scale):
    assert py_func(n_bytes, window, rounds, seed, key_scale) == cy_func(
        n_bytes, window, rounds, seed, key_scale
    )
