"""Soft-clip threshold checks for SAM CIGAR strings."""

from __future__ import annotations

import re

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def split_cigar(cigar: str) -> tuple[list[int], list[str]]:
    lengths: list[int] = []
    ops: list[str] = []
    for n, op in _CIGAR_RE.findall(cigar):
        lengths.append(int(n))
        ops.append(op)
    return lengths, ops


def softclipped_bases(cigar: str) -> int:
    lengths, ops = split_cigar(cigar)
    return sum(n for n, op in zip(lengths, ops, strict=False) if op == "S")


def check_aln_softclip(
    cigar: str, query_length: int, single_end_thresh: float, is_unmapped: bool = False
) -> tuple[bool, int]:
    clip = softclipped_bases(cigar)
    single_limit = single_end_thresh * query_length
    return (clip < single_limit and not is_unmapped), clip


def check_pair_clips(
    cigar1: str,
    len1: int,
    cigar2: str,
    len2: int,
    template_length: int,
    single_end_thresh: float,
    both_end_thresh: float,
) -> bool:
    read1_ok, clip1 = check_aln_softclip(cigar1, len1, single_end_thresh)
    read2_ok, clip2 = check_aln_softclip(cigar2, len2, single_end_thresh)
    total = clip1 + clip2

    if template_length > 0:
        pair_ok = total < both_end_thresh * abs(template_length)
    else:
        t1 = both_end_thresh * len1
        t2 = both_end_thresh * len2
        pair_ok = (total < t1) and (total < t2)

    return read1_ok and read2_ok and pair_ok
