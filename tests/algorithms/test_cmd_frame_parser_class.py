"""Test cmd_frame_parser_class equivalence."""

import pytest

from cnake_charmer.cy.algorithms.cmd_frame_parser_class import cmd_frame_parser_class as cy_func
from cnake_charmer.py.algorithms.cmd_frame_parser_class import cmd_frame_parser_class as py_func


@pytest.mark.parametrize(
    "n_frames,payload_len,seed,mod", [(90, 11, 7, 127), (220, 17, 13, 255), (300, 23, 19, 511)]
)
def test_cmd_frame_parser_class_equivalence(n_frames, payload_len, seed, mod):
    assert py_func(n_frames, payload_len, seed, mod) == cy_func(n_frames, payload_len, seed, mod)
