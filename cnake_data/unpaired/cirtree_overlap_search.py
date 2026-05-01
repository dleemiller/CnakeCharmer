"""CIR-tree overlap helpers and recursive block search."""

from __future__ import annotations


def ovcmp(a_hi: int, a_lo: int, b_hi: int, b_lo: int) -> int:
    if a_hi < b_hi:
        return 1
    if a_hi > b_hi:
        return -1
    if a_lo < b_lo:
        return 1
    if a_lo > b_lo:
        return -1
    return 0


def overlaps(
    qchrom: int,
    qstart: int,
    qend: int,
    rstartchrom: int,
    rstartbase: int,
    rendchrom: int,
    rendbase: int,
) -> bool:
    return (
        ovcmp(qchrom, qstart, rendchrom, rendbase) > 0
        and ovcmp(qchrom, qend, rstartchrom, rstartbase) < 0
    )


class CIRTreeIndex:
    def __init__(self, reader, root_offset: int):
        self.reader = reader
        self.root_offset = root_offset

    def _find_overlapping(
        self,
        index_file_offset: int,
        chrom_ix: int,
        start: int,
        end: int,
        out: list[tuple[int, int]],
    ) -> None:
        self.reader.seek(index_file_offset)
        is_leaf, _, child_count = self.reader.read_and_unpack("BBH", 4)
        if is_leaf:
            self._find_overlapping_leaf(chrom_ix, start, end, out, child_count)
        else:
            self._find_overlapping_parent(chrom_ix, start, end, out, child_count)

    def _find_overlapping_leaf(
        self, chrom_ix: int, start: int, end: int, out: list[tuple[int, int]], child_count: int
    ) -> None:
        arr = self.reader.read_and_unpack("LLLLQQ" * child_count, 32 * child_count)
        for i in range(child_count):
            s_chrom, s_base, e_chrom, e_base, offset, size = arr[i * 6 : i * 6 + 6]
            if overlaps(chrom_ix, start, end, s_chrom, s_base, e_chrom, e_base):
                out.append((offset, size))

    def _find_overlapping_parent(
        self, chrom_ix: int, start: int, end: int, out: list[tuple[int, int]], child_count: int
    ) -> None:
        arr = self.reader.read_and_unpack("LLLLQ" * child_count, 24 * child_count)
        for i in range(child_count):
            s_chrom, s_base, e_chrom, e_base, child_offset = arr[i * 5 : i * 5 + 5]
            if overlaps(chrom_ix, start, end, s_chrom, s_base, e_chrom, e_base):
                self._find_overlapping(child_offset, chrom_ix, start, end, out)

    def find_overlapping_blocks(self, chrom_ix: int, start: int, end: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        self._find_overlapping(self.root_offset, chrom_ix, start, end, out)
        return out
