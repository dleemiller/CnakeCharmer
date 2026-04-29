"""Minimal FASTQ sequence record and iterator parser."""

from __future__ import annotations


class FormatError(Exception):
    pass


class Sequence:
    def __init__(self, name, sequence, qualities=None, twoheaders=False):
        self.name = name
        self.sequence = sequence
        self.qualities = qualities
        self.twoheaders = twoheaders
        if qualities is not None and len(qualities) != len(sequence):
            raise FormatError("quality length does not match sequence length")

    def __len__(self):
        return len(self.sequence)


class FastqReader:
    def __init__(self, fp):
        self.fp = fp

    def __iter__(self):
        it = iter(self.fp)
        i = 0
        name = sequence = qualities = None
        twoheaders = False
        for line in it:
            if i == 0:
                if not line.startswith("@"):
                    raise FormatError("expected '@'")
                name = line[1:].rstrip("\r\n")
            elif i == 1:
                sequence = line.rstrip("\r\n")
            elif i == 2:
                if not line.startswith("+"):
                    raise FormatError("expected '+'")
                hdr = line.rstrip("\r\n")
                twoheaders = len(hdr) > 1
                if twoheaders and hdr[1:] != name:
                    raise FormatError("mismatched second header")
            else:
                qualities = line.rstrip("\r\n")
                yield Sequence(name, sequence, qualities, twoheaders=twoheaders)
            i = (i + 1) % 4
        if i != 0:
            raise FormatError("FASTQ ended prematurely")
